"""
The Netherlands (Dutch) flag is three stripes of red, white, and blue. Given red, white, and blue balls 
randomly arranged in a line, arrange them so all balls of the same color are together and their
collective color groups are in the correct order.
"""

(red, white, blue) = (0, 1, 2)

def dfp(pivot=0, a):
    """
    For some array of balls a (as described above), partition into the Dutch flag.
    We rely on the pivot index of the array a.
    """
