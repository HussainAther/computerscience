import tkinter
import random

"""
In this game player has to enter color of the word that appears on the screen and hence the score increases 
by one, the total time to play this game is 30 seconds. Colors used in this game are Red, Blue, Green, Pink, 
Black, Yellow, Orange, White, Purple and Brown. Interface will display name of different colors in different 
colors. Player has to identify the color and enter the correct color name to win the game.
"""

colors = ["Red","Blue","Green","Pink","Black", 
          "Yellow","Orange","White","Purple","Brown"] 
score = 0

timeleft = 30

def startGame(event):
    """
    Start the game with a given event.
    """
