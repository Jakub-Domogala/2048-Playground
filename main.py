from models.Board import Board
from time import sleep
import random

def main():
    board = Board(4)
    print(board.add_num())
    print(board.add_num())
    while not board.is_over():
        if board.make_move_in_dir(random.randint(0, 3)):
            board.add_num()
        print(board)
        print('Score:', board.get_current_score())
        print()
        sleep(0.1)
    print('GAME OVER')


if __name__ == "__main__":
    main()