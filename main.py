from models.Board import Board
from time import sleep

def main():
    board = Board(4)
    print(board.add_num())
    print(board.merge_nums([2,2,2,4,4]))
    while True:
        board.make_move_in_dir(0)
        board.add_num()
        board.print()
        print()
        sleep(1)
        board.make_move_in_dir(2)
        board.add_num()
        board.print()
        print()
        sleep(1)
        print(board.get_current_score())


if __name__ == "__main__":
    main()