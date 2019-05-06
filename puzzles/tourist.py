"""
Manhattan tourist problem (Tourist).

Find the length of a longest path between two points
in a matrix.
"""

def mantour(n, m, d, r):
    """
    For two integers n and m, an n * (m+1) matrix down
    and an (n+1)*m matrix right describing the grid,
    return the length of a longest path from the source (0, 0) 
    to the sink (n, m) in the grid n * m the edgdes of the matrices
    down and right.
    """
    
