"""
Longest common subsequence between two sequences.
"""

def lcslength(x, y)
    """
    For two sequences x and y, store c[i, j] values in a table. Compute the 
    entries in row-major order. Maintain the table to construct an optimal 
    solution.
    """
    m = len(x)
    n = len(y)
    b, c = [[]], [[]] # output table lists
    for i in range(m):
        c[i][0] = 0
    for j in range(n):
        c[0][j] = 0
    for i in range(1, m):
        for j in range(1, n):
            if x[i] == y[j]:
                c[i][j] = c[i-1][j-1] + 1
                b[i][j] = "upleft"
            elif c[i-1][j] >= c[i][j-1]:
                c[i][j] = c[i-1][j]
                b[i][j] = "up"
            else:
                c[i][j] = c[i][j-1]
                b[i][j] = "left"
    return b, c 
