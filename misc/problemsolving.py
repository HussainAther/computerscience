import fileinput
import re
import sys

"""
Problem solving functionality.
"""

class World:
    """
    Create a world we can achieve.
    """
    def __init__(self):
        self.state = dict()
        self.goals = set()
        self.known_literals = set()
        self.actions = dict()
