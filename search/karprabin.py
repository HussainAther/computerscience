
"""
The Karp-Rabin algorithm matches a pattern to a string, the basic principle of database search.

It computes scores of the substrings against the main string. It compares each position of the
substrings against the main string.
"""

def kr(p, t, q):
    """
    Search for patterns p within a string t for some prime number q.
    """
    i = 0
    j = 0
    p = 0    # hash value for pattern
    t = 0    # hash value for txt
    h = 1

    for i in range(len(p)-1):
        h = (h * d)% q
    
    for i in range(len(p)):
        a = (d * p + ord(p[i]))% q
        b = (d * t + ord(t[i]))% q

    for i in range(len(t) - len(p) + 1):
        if a == b:
            for j in range(len(p)):
                if b[i + j] != a[j]:
                    break
            j+= 1
            if j == len(p):
                print("Pattern found at index : %f"  % str(i))

        if i < len(t) - len(p):
            b = (d*(b-ord(t[i])*h) + ord(t[i + len(p)])) % q
            if b < 0:
                b = b + q

