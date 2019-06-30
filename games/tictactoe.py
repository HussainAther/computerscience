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
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)
        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end
        self.end = False
        return self.end
