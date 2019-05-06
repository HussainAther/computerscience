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

    # Initialize the distance dictionary
    # with the first entry being the origin with 
    # distance 0 
    s = {(0,0):0}
   
    # For each row, enter distance in the first column
    # using the down matrix 
    for i in range(1, n+1):
        s[(i,0)] = s[(i-1,0)] + d[i-1][0]
       
    # For each column, enter distance in the first row 
    # using the right matrix
    for j in range(1, m+1):
        s[(0,j)] = s[(0,j-1)] + r[0][j-1]

    # Calculate the distances
    for i in range(1,n+1):
        for j in range(1,m+1):
            s(i,j)] = max(s[(i-1,j)] + d[i-1][j], s[(i,j-1)] + r[i][j-1])

    return s[(n,m)]  
