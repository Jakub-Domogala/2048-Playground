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
        
        # TODO
        # self.observation_space = 

        self.reset()

    def reset(self):
        self.board = Board(self.size)
        self.board.add_num()
        self.board.add_num()
        return self._get_observation()

    def _get_observation(self):
        # TODO
        pass

    def step(self, action):
        self.board.make_move_in_dir(action)
        obs = self._get_observation()
        reward = self.board.get_current_score()
        done = self.board.is_over()
        
        return obs, reward, done, {}

    def render(self, mode='human'):
        print(self.board)
