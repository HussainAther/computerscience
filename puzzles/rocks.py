"""
This is a game between two players. It starts with two piles of rocks, n and m rocks per pile. On each turn, a
player may take one rock from one pile, or two rocks, one from each pile. If one pile is empty, you can
only take one rock from the other pile. The player who takes the last rock or rocks wins the game.
"""

def rocks(n, m):
    """
    For integer piles n and m, check if the player loses.
    """
    if n % 2 == 0 and m % 2 == 0:
        return "lose"
    return "win"
