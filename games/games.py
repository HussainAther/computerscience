from utils import *
import random

"""
Python implementation of vairous search processes in games.
"""

def minimax_decision(state, game):
    """
    Minimax search. Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states.
    """
    player = game.to_move(state)

