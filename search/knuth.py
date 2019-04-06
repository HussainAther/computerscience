"""
With the Knuth-Morris-Pratt matching algorithm (KMP) we follow
a finite-automatan-matcher (we can call KMP-Matcher) that calls the
auxiliary procedure compute-prefix-function (cpf) to compute pi.
"""

def cpf(a):
    """
    Compute-prefix-function (cpf) to calclate pi as a list. 
    """
    P = list(a)
    m = len(a)
    a = [0] * m
    k = 0
    for q in range(2, m + 1):
        while k > 0 and P[k] != P[q - 1]:
            k = a[k - 1]
        if P[k] == P[q - 1]:
            k += 1
        a[q - 1] = k
    return a

def kmp(t, p):
    """
    Match text t against p using the Knuth-Morris-Pratt matching algorithm.
    """
    n = len(t)
    m = len(p)
    pi = cpf(p)
    q = 0 # number of matched characters
    for i in range(n): # scan the text 
        while q > 0 and p[q+1] != t[i]:
            q = pi[q]
        if p[q+1] == t[i]:
            q += 1
        if q == m:
            print("Shifted pattern %" % (i-m))
            q = pi[q]
     

         
