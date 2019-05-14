"""
A king stands on the upper left square of the chessboard. Two players make turns 
moving the king either one square to the right or one square downward or one 
square along a diagonal in the southeast direction. The player who can place the 
king on the lower right square of the chessboard wins. Who will win? Describe the winning strategy.
"""

def king(p)
    """
    For some starting tuple position p, determine if player one will win.
    
    We notice that player one can force player two to lose moving
    directly to position g2, right above the bottom-right square.
    At f3, player one also will lose. Moving either right or down 
    by one or two lets player two win. This continues until the 
    top-left so that, for plyaer two to win, player two should
    make his/her first move by moving the king to the diagnol. This
    means player one will move into a losing position for the next turn. 
    """
    
