"""
McCreight's algorithm (McCreight mccreight) builds a suffix tree in linear time.
"""

def mccreight(a):
    """
    For an input list of suffixes a, build a suffix tree that repressents
    all possible suffixes of a.
    """
    t = ["root"] # start with root node
     
