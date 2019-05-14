"""
A queen stands on the third square of the uppermost row of a chessboard. Two 
players take turns moving the queen either horizontally to the right or vertically 
downward or diagonally in the southeast direction (as many squares as they want). 
The player who can place the queen on the lower right square of the chessboard 
wins. Who will win? Describe the winning strategy.
"""

def queen():
    """
    Player one will win.
    When player two moves the queen onto the diagnol or to the farthest
    right column, player one wins. For player one to win, he/she must move 
    the queen three positions to the right first. This way, player two is either 
    forced to move to the diagnol, move to the farthest right column, move to
    the second-farthest right column, or move downward in such a way that 
    player one won't be the one to move onto the diagnol. If player two chooses
    to move to the square one above the diagnol, player one may just move diagnolly 
    down-right one position. This forces player two to move either onto the diagnol or
    to the far-right column.  
    """
