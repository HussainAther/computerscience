import numpy as np
import pickle

"""
Tic-Tac-Toe implementation using reinforcement learning.
"""

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

class State:
    """
    The board is represented by an n * n array,
    1 represents a chessman of the player who moves first,
    -1 represents a chessman of another player.
    0 represents an empty position.
    """
    def __init__(self):
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None
        self.end = None:w

    def hash(self):
        """
        Compute the hash value for one state.
        """
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val

    def is_end(self):
        """
        Check if a player has won.
        """
        if self.end is not None:
            return self.end
        results = []
        # check row
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # check columns
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))
