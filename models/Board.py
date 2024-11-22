from random import choice
import numpy as np
import time

class Board():
    def __init__(self, size):
        self.size = size
        self.board = np.full((size, size), None, dtype=object)
        self.score = 0
        self.moves_count = 0
        self.moves = [(1,0), (0,1), (-1,0), (0,-1)]

    def print(self):
        for row in self.board:
            formatted_row = [f"{num:3}," if num is not None else "   ," for num in row]
            print("".join(formatted_row))
    
    def get_list_of_empty(self):
        return [(i, j) for i in range(self.board.shape[0]) for j in range(self.board.shape[1]) if self.board[j, i] is None]
        # return np.argwhere(self.board == None)
    
    def get_current_score(self):
        # TODO score should be calculated in totally diffrent way xd
        # pretty important for learning
        return self.score
        # return np.sum(np.where(self.board == None, 0, self.board))

    def is_lost(self):
        return 0 == sum([self.is_movable_in_dir(i) for i in range(self.size)])

    def add_num(self):
        """
        Adds new 2 to the Board and return True if successful, 
        False if Board was already full
        """
        options = self.get_list_of_empty()
        print(options)
        if len(options) == 0:
            return False
        x, y = choice(options)
        self.board[y][x] = 2
        return True

    
    def merge_nums(self, nums, add_to_score = False):
        prev = None
        merged = []
        for n in nums:
            if n == None:
                continue
            if n == prev:
                merged[-1] = n + n
                if add_to_score:
                    self.score += merged[-1]
                prev = None
            else:
                merged.append(n)
                prev = n
        return merged + [None] * (self.size - len(merged))

    
    def make_move_in_dir(self, id):
        """
        Makes a move of given id, returns True if successful,
        False if given move was not possible
        """
        is_moved = False
        if id == 1 or id == 3:
            direction = -1 if id == 1 else 1
            for i in range(self.size):
                nums = self.board[:,i][::direction]
                merged_nums = self.merge_nums(nums, True)
                if not np.array_equal(nums, merged_nums):
                    is_moved = True
                    self.board[:,i] = merged_nums[::direction]
        elif id == 0 or id == 2:
            direction = -1 if id == 0 else 1
            for i in range(self.size):
                nums = self.board[i][::direction]
                merged_nums = self.merge_nums(nums, True)
                if not np.array_equal(nums, merged_nums):
                    is_moved = True
                    self.board[i] = merged_nums[::direction]
        return is_moved
    
    def is_movable_in_dir(self, id):
        """
        Return True if move in given dir is possible
        else False
        """
        is_moved = False
        if id == 1 or id == 3:
            direction = -1 if id == 1 else 1
            for i in range(self.size):
                nums = self.board[:,i][::direction]
                merged_nums = self.merge_nums(nums)
                if not np.array_equal(nums, merged_nums):
                    is_moved = True
        elif id == 0 or id == 2:
            direction = -1 if id == 0 else 1
            for i in range(self.size):
                nums = self.board[i][::direction]
                merged_nums = self.merge_nums(nums)
                if not np.array_equal(nums, merged_nums):
                    is_moved = True
        return is_moved
    
    def play_random(self):
        ids = list(range(len(self.moves)))
        self.add_num()
        self.print()
        while not self.is_lost():
            next_move = choice(ids)
            while not self.is_movable_in_dir(next_move):
                next_move = choice(ids)
            self.make_move_in_dir(next_move)
            self.print()
            print(self.get_current_score())
            self.add_num()
            self.print()
            # time.sleep(0.2)
        print(f"You lost with score of {self.get_current_score()}")