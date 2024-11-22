from models.Board import Board
from time import sleep

def main():
    print(sum([False, True]))
    board = Board(4)
    board.play_random()


if __name__ == "__main__":
    main()