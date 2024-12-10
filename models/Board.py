from random import choice
import numpy as np

class Board():
    def __init__(self, size):
        self.size = size
        self.board = np.full((size, size), 0)
        self.score = 0
        self.moves_count = 0
        self.moves = [(1,0), (0,1), (-1,0), (0,-1)]
        self.score = 0
            
    def __str__(self):
        cell_width = 5
        create_sep = lambda l, c, r:  l + c.join("─" * cell_width for _ in self.board[0]) + r + '\n'
        
        separator = create_sep('├', '┼', '┤')
        upper_bar = create_sep('╭', '┬', '╮')
        lower_bar = create_sep('╰', '┴', '╯')

        rows = "".join([
            x for row in self.board for x in (("│" + "│".join(f"{str(cell or ''):^{cell_width}}" for cell in row) + "│\n"), separator)
        ][:-1])

        return f"{upper_bar}{rows}{lower_bar}"
    
    def get_list_of_empty(self):
        return np.argwhere(self.board == 0)
    
    def get_current_score(self):
        # TODO score should be calculated in totally different way xd
        # pretty important for learning
        return self.score

    def add_num(self):
        """
        Adds new 2 to the Board and return True if successful, 
        False if Board was already full
        """
        options = self.get_list_of_empty()
        if len(options) == 0:
            return False
        row, col = choice(options)
        self.board[row, col] = 2
        return True

    
    def merge_nums(self, nums):
        added_score = 0
        prev = None
        merged = []
        for n in nums:
            if n == 0:
                continue
            if n == prev:
                merged[-1] = n + n
                added_score += n + n
                prev = None
            else:
                merged.append(n)
                prev = n
        return merged + [0] * (self.size - len(merged)), added_score

    
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
                merged_nums, added_score = self.merge_nums(nums)
                if not np.array_equal(nums, merged_nums):
                    is_moved = True
                    self.board[:,i] = merged_nums[::direction]
                    self.score += added_score
        elif id == 0 or id == 2:
            direction = -1 if id == 0 else 1
            for i in range(self.size):
                nums = self.board[i][::direction]
                merged_nums, added_score = self.merge_nums(nums)
                if not np.array_equal(nums, merged_nums):
                    is_moved = True
                    self.board[i] = merged_nums[::direction]
                    self.score += added_score
        
        return is_moved
    
    def is_over(self):
        for move in range(4):
            if self.is_movable_in_dir(move):
                return False
        return True
    
    def get_max_tile(self):
        return np.max(self.board)
    
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
                merged_nums, _ = self.merge_nums(nums)
                if not np.array_equal(nums, merged_nums):
                    is_moved = True
        elif id == 0 or id == 2:
            direction = -1 if id == 0 else 1
            for i in range(self.size):
                nums = self.board[i][::direction]
                merged_nums, _ = self.merge_nums(nums)
                if not np.array_equal(nums, merged_nums):
                    is_moved = True
        return is_moved