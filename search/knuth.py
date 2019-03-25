"""
With the Knuth-Morris-Pratt matching algorithm (KMP) we follow
a finite-automatan-matcher (we can call KMP-Matcher) that calls the
auxiliary procedure compute-prefix-function (cpf) to compute pi.
"""

def cpf(pattern):
    """
    Compute-prefix-function (cmp) 
    """
    P = list(pattern)
    m = len(pattern)
    a = [0] * m
    k = 0
    for q in range(2, m + 1):
        while k > 0 and P[k] != P[q - 1]:
            k = a[k - 1]
        if P[k] == P[q - 1]:
            k += 1
        a[q - 1] = k
    return a

def kmp(
