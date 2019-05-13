"""
McCreight's algorithm (McCreight mccreight) builds a suffix tree in linear time.
"""

def mccreight(a):
    """
    For an input list of suffixes a, build a suffix tree that repressents
    all possible suffixes of a.
    """
    t = {0: "root"} # start with root node. key for level. value for node.
    for i, j in enumerate(a):
        d = len(t) # depth
        t[d+1] = i # add the first node
        t[d+1] = j # add the second node
