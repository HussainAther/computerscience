"""
Hoare's (Hoare) partition scheme uses two indices that start at
the neds of the array partitioned and move toward one another until
they detect an inversion. The inverted elements are swapped.  
"""
def qs(a):
    """
    Quicksort algorithm
    """
    if a[0] < a[-1]:
        p = hoare(a, a[0], a[-1])
        qs(a, a[0], p)
        qs(a, p+1, a[-1])


def hoare(a, l, h):
    """
    Hoare's algorithm. Partition using swaps for an array a with first value l
    and final value h.
    """
    piv = a[(l + h)/2] # pivot value
    i = l - 1
    j = h + 1
    while True: # loop indefinitely 
        i += 1
        while a[i] < piv:
            j -= i
        if i >= j:
            return j
        (a[i], a[j]) = (a[j], a[i])
        
