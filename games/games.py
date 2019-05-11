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
    def max_value(state, alpha, beta, depth):
        """
        Return the maximum value of a state.
        """
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -infinity
        for (a, s) in game.successors(state):
            v = max(v, min_value(s, alpha, beta, depth+1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

   def min_value(state, alpha, beta, depth):
        """
        Return the minimum value of a state.
        """
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = infinity
        for (a, s) in game.successors(state):
            v = min(v, max_value(s, alpha, beta, depth+1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    cutoff_test = (cutoff_test or
                   (lambda state,depth: depth>d or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    action, state = argmax(game.successors(state),
                           lambda ((a, s)): min_value(s, -infinity, infinity, 0))
    return action

def query_player(game, state):
    """"
    Make a move by querying standard input.
    """
    game.display(state)
    return num_or_str(raw_input('Your move? '))

def random_player(game, state):
    """
    A player that chooses a legal move at random.
    """
    return random.choice(game.legal_moves())

def alphabeta_player(game, state):
    """
    Perform the alpha beta search for a state and game.
    """
    return alphabeta_search(state, game)

def play_game(game, *players):
    """
    Play an n-person, move-alternating game.
    """
    state = game.initial
    while True:
        for player in players:
            move = player(game, state)
            state = game.make_move(move, state)
            if game.terminal_test(state):
                return game.utility(state, players[0])

class Game:
    """
    A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement
    legal_moves, make_move, utility, and terminal_test. You may
    override display and successors or you can inherit their default
    methods. You will also need to set the .initial attribute to the
    initial state; this can be done in the constructor.
    """
   def legal_moves(self, state):
        """
        Return a list of the allowable moves at this point.
        """"
	return ["l", "r", "u", "d"] 

    def make_move(self, move, state):
        """
        Return the state that results from making a move from a state.
        """
        action, state = argmax(game.successors(state),
                           lambda ((a, s)): min_value(s, -infinity, infinity, 0))
        return state

    def utility(self, state, player):
        """
        Return the value of this final state to player.
        """
        return state 

    def terminal_test(self, state):
        """
        Return True if this is a final state for the game.
        """
        return not self.legal_moves(state)

    def to_move(self, state):
        """
        Return the player whose move it is in this state.
        """
        return state.to_move

    def display(self, state):
        """
        Print or otherwise display the state.
        """"
        print(state)

    def successors(self, state):
        """
        Return a list of legal (move, state) pairs.
        """
        return [(move, self.make_move(move, state))
                for move in self.legal_moves(state)]

    def __repr__(self):
        return "<%s>" % self.__class__.__name__

class TicTacToe(Game):
    """
    Play TicTacToe on an h x v board, with Max (first player) playing "X".
    A state has the player to move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is "X" or "O".
     """
    def __init__(self, h=3, v=3, k=3):
        update(self, h=h, v=v, k=k)
        moves = [(x, y) for x in range(1, h+1)
                 for y in range(1, v+1)]
        self.initial = Struct(to_move='X', utility=0, board={}, moves=moves)
