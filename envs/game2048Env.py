import gym
from gym import spaces
import numpy as np
from models.Board import Board

class Game2048Env(gym.Env):
    def __init__(self, size=4):
        super(Game2048Env, self).__init__()
        self.size = size
        self.board = Board(size)
        
        # Action space: 4 discrete actions (up, right, down, left)
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(0, 1, [size,size,16], dtype=int)

        self.reset()

    def reset(self):
        self.board = Board(self.size)
        self.board.add_num()
        self.board.add_num()
        return self._get_observation()

    def _get_observation(self):
        log_board = np.zeros_like(self.board.board, dtype=int)
        non_zero_mask = self.board.board > 0
        log_board[non_zero_mask] = np.log2(self.board.board[non_zero_mask]).astype(int)
        return np.eye(16, dtype=int)[log_board].reshape(4, 4, 16).transpose(2, 0, 1)
    
    
    def step(self, action):
        prev_score = self.board.get_current_score()
        prev_max_tile = self.board.get_max_tile()

        is_moved = self.board.make_move_in_dir(action)
        self.board.add_num()

        obs = self._get_observation()
        
        reward = self.board.get_current_score() - prev_score
        if not is_moved:
            reward -= 0.1  # Penalize invalid moves
        max_tile = self.board.get_max_tile()
        if max_tile > prev_max_tile:
            reward += max_tile / 100.0  # Bonus for creating larger tiles
        
        reward /= 1.0 # Normalize reward for stable training
        
        done = self.board.is_over()
        
        return obs, reward, done, {}

    def render(self, mode='human'):
        print(self.board)
