"""
Find the longest common substring between two strings.

subsequence sequence
"""

def lcs(x, y):
    """
    Return the longest common substring (subsequence)
    between two strings x and y.
    """
    result = ""
    m, n = len(x, len(y)
    for i in range(m):
        match = ""
        for j in range(n):
            if (i + j < m and x[i + j] == y[j]):
                match += y[j]
            else:
                if (len(match) > len(result)): result = match
                match = ""
    return result

def lcsl(x, y): 
    """
    Return the length of the longest common substring.
    Use a table to store lengths of the longest common
    suffixes of substrings. The longest common suffix of 
    x and y. The first row and first column entries have
    no logical meaning.
    """
    m, n = len(x), len(y)
    suff = [[0 for k in range(n+1)] for l in range(m+1)] # suffix table 
    result = 0 
    for i in range(m+1): 
        for j in range(n+1): 
            if (i == 0 or j == 0): 
                suff[i][j] = 0
            elif (x[i-1] == y[j-1]): 
                suff[i][j] = suff[i-1][j-1] + 1
                result = max(result, suff[i][j]) 
            else: 
                suff[i][j] = 0
    return result
    
"""
Find the shortest superstring (supersequence).
"""

def super(x, y):
    """
    Shortest superstring between strings x and y.
    Use longest common substring to determine how 
    a superstring results.
    """
    lc = lcs(x, y)
    scs = "" # shortest common superstring 
    while len(lc) > 0:
        if y[0] == lc[0] and y[0] == lc[0]:
            scs += lcs[0]
            lc = lc[1:]
            x = x[1:]
            y = y[1:]
        elif x[0] == lc[0]:
            scs += y[0]
            y = by[1:]
        else:
            scs += x[0]
            x = x[1:]
    return scs + x + y
