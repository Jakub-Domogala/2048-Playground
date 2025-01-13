import sys
import numpy as np
import math
import random
from envs.game2048Env import Game2048Env

import gym
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from collections import deque
from dataclasses import dataclass
import os
import yaml
import argparse
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self, input_units=16, depth1=128, depth2=256, hidden_units=512, output_units=4):
        super(DQN, self).__init__()
        # Convolutional layers (Layer 1)
        self.conv1_layer1 = nn.Conv2d(input_units, depth1, kernel_size=(1, 2), stride=1, padding=0)
        self.conv2_layer1 = nn.Conv2d(input_units, depth1, kernel_size=(2, 1), stride=1, padding=0)

        # Convolutional layers (Layer 2)
        self.conv1_layer2 = nn.Conv2d(depth1, depth2, kernel_size=(1, 2), stride=1, padding=0)
        self.conv2_layer2 = nn.Conv2d(depth1, depth2, kernel_size=(2, 1), stride=1, padding=0)

        # Fully connected layers
        expand_size = (2 * 4 * depth2 * 2 + 3 * 3 * depth2 * 2 + 4 * 3 * depth1 * 2)
        self.fc1 = nn.Linear(expand_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_units)

    def forward(self, x):
        # Layer 1
        conv1 = F.relu(self.conv1_layer1(x))
        conv2 = F.relu(self.conv2_layer1(x))

        # Layer 2
        conv11 = F.relu(self.conv1_layer2(conv1))
        conv12 = F.relu(self.conv2_layer2(conv1))
        conv21 = F.relu(self.conv1_layer2(conv2))
        conv22 = F.relu(self.conv2_layer2(conv2))

        # Flatten activations
        hidden1 = conv1.view(conv1.size(0), -1)
        hidden2 = conv2.view(conv2.size(0), -1)
        hidden11 = conv11.view(conv11.size(0), -1)
        hidden12 = conv12.view(conv12.size(0), -1)
        hidden21 = conv21.view(conv21.size(0), -1)
        hidden22 = conv22.view(conv22.size(0), -1)

        # Concatenate all hidden layers
        hidden = torch.cat([hidden1, hidden2, hidden11, hidden12, hidden21, hidden22], dim=1)

        # Fully connected layers
        hidden = F.relu(self.fc1(hidden))
        output = self.fc2(hidden)
        return output
    
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1

    # Algorithm specific arguments
    episodes: int = 1000
    learning_rate: float = 1e-5
    buffer_size: int = 1000000
    gamma: float = 0.99
    target_update_freq: int = 20
    batch_size: int = 32
    start_epsilon: float = 1.0
    min_epsilon: float = 0.01
    epsilon_decay: float = 0.997
    max_try: int = 20000
    
    @staticmethod
    def from_yaml(file_path: str):
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return Args(**config)


class Learner():
    def __init__(self, args: Args):
        self.args = args
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = Game2048Env(size=4)
        self.q_network = DQN().to(device)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=args.learning_rate, amsgrad=True)
        # self.optimizer = torch.optim.RMSprop(self.q_network.parameters(), lr=args.learning_rate, alpha=0.99, eps=1e-6)
        self.target_network = DQN().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.target_network.eval()
        self.q_network.train()

        self.replay_buffer = deque(maxlen=args.buffer_size)
        self.epsilon = self.args.start_epsilon


    def optimize_model(self):
        if len(self.replay_buffer) < self.args.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.args.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch))
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch))
        done_batch = torch.FloatTensor(done_batch)

        # Compute Q-values for current states
        q_values = self.q_network(state_batch).gather(1, action_batch)

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.args.gamma * max_next_q_values * (1 - done_batch)

        loss = nn.SmoothL1Loss()(q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()
        return loss
        
    def sample_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            q_values = self.q_network(torch.FloatTensor(state).unsqueeze(0))
            action = torch.argmax(q_values).item()
        return action
    
    def learn(self):        
        self.rewards_per_episode = []
        self.epsilon = self.args.start_epsilon

        for episode in tqdm(range(self.args.episodes), desc="Training Progress", leave=True):
            
            state = self.env.reset()
            episode_reward = 0
            
            for t in range(self.args.max_try):
                action = learner.sample_action(state)
                next_state, reward, done, _ = self.env.step(action)
                learner.replay_buffer.append((state, action, reward, next_state, done))
                
                loss = learner.optimize_model()
                episode_reward += reward
                state = next_state
                
                if done:
                    break
                
            tqdm.write(f"Episode {episode + 1}: Moves = {t}, Max tile = {self.env.board.get_max_tile()}, Reward = {episode_reward:.2f}, Loss = {loss:.5f}")
            self.rewards_per_episode.append(episode_reward)
        
            if episode % self.args.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.q_network.train()
                self.save_rewards_and_parameters()
                self.save_checkpoint(episode, loss)
                
            
            self.epsilon = max(self.args.min_epsilon, self.epsilon * self.args.epsilon_decay)
        self.save_checkpoint(episode, loss)
    
    def save_rewards_and_parameters(self, filename=None):
        if filename is None:
            filename = self.args.exp_name
        np.savetxt(f'results/{filename}', [self.rewards_per_episode])
        
        with open((f"parameters/{filename}-config.yaml"), "w") as f:
            yaml.dump(args.__dict__, f)

    
    def save_checkpoint(self, epoch, loss, filename="checkpoint.pth"):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file (YAML).", default='config.yaml')
    args = parser.parse_args()

    args = Args.from_yaml(args.config)

    
    learner = Learner(args)
    learner.learn()
    learner.save_rewards_and_parameters()

    