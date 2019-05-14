"""
Find the longest common substring between two strings.

subsequence sequence
"""

def lcs(x, y): 
    """
    Use a table to store lengths of the longest common
    suffixes of substrings. The longest common suffix of 
    x and y. The first row and first column entries have
    no logical meaning.
    """
    m = len(x)
    n = len(y)
    suff = [[0 for k in range(n+1)] for l in range(m+1)] # suffix 
    result = 0 
    for i in range(m + 1): 
        for j in range(n + 1): 
            if (i == 0 or j == 0): 
                suff[i][j] = 0
            elif (x[i-1] == y[j-1]): 
                suff[i][j] = suff[i-1][j-1] + 1
                result = max(result, suff[i][j]) 
            else: 
                sufff[i][j] = 0
    return result 
