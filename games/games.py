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
    
    def max_value(state):
        """
        Return the maximum value of a state.
        """
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for (a, s) in game.successors(state):
            v = max(v, min_value(s))
        return v
    
    def min_value(state):
        """
        Return the minimum value of a state.
        """
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for (a, s) in game.successors(state):
            v = min(v, max_value(s))
        return v
