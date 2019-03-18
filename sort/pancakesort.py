"""
Pancake sort algorithm for various parts of the list
until they're sorted.
"""

def f(a, i):
    # flip the array
    s = 0
    while s < i:
        t = a[s]
        a[s] = a[i]
        a[i] = t
        s += 1
        i -= 1

def pancakeSort(a, n):
    """
    Goro Akechi ain't got nothin' on this.
    """
    c = n
    while c > 1:
        mi = findMax(a, c)
        if mi != c-1:
            f(a, mi)
            f(a, c-1)
        c -= 1
