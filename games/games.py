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

    action, state = argmax(game.successors(state),
                           lambda ((a, s)): min_value(s))
    return action

def alphabeta_full_search(state, game):
    """
    Search game to determine best action. Use alpha-beta pruning.
    Alpha beta Alpha-Beta Beta beta. 
    This version searches all the way to the leaves.
    """

    player = game.to_move(state)
    
    def max_value(state, alpha, beta):
        """
        Return the maximum state value for given alpha and beta values.
        """
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for (a, s) in game.successors(state):
            v = max(v, min_value(s, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def min_value(state, alpha, beta):
        """
        Return the minimum state value for given alpha and beta values. 
        """
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for (a, s) in game.successors(state):
            v = min(v, max_value(s, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    action, state = argmax(game.successors(state),
                           lambda ((a, s)): min_value(s, -infinity, infinity))
    return action


def alphabeta_search(state, game, d=4, cutoff_test=None, eval_fn=None):
    """
    Search game to determine best action; use alpha-beta pruning.
    Alpha beta Alpha-Beta Beta beta. 
    This version cuts off search and uses an evaluation function.
    """

    player = game.to_move(state)
