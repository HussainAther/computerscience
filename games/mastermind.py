import random

"""
Python implementation of Mastermind, a puzzle game
in which the user must guess a 4-character pattern
based on feedback from a computer.
"""

def encode(c, g):
    """
    Return the correct c based on guess g of the user.
    """
    out = [""] * len(c) # output array
    for i, (cc, gc) in enumerate(zip(c, g)): # for each correct character
                                             # and guess character
 
    return "".join(out)
