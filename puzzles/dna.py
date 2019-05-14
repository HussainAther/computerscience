"""
Assume you have a DNA base sequence P of n bases and are given
a sequence S of m bases. The problem is to find the place in S where P first
occurs, if there is one.
"""

def naive(P, S):
    """
    Naïve exhaustive search (naive Naive) algorithm for P sequence
    and S sequence. Find the location i or output False otherwise.
    """
    n = len(P)
    m = len(S)
    for i in range(n-m+1):
        for j in range(m):
            if S[i+j] != P[j]:
                break
            if j == len(m):
                return i
    return False 
