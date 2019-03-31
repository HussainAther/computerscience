import sys

"""
Return a tuple of the indices demarcating a maximum subarray 
of some array a that crosses the midpoint along with the sum
fo the values in a maximum subarray.
"""

def fmcs(a):
    """
    Find max crossing subarray for an array a.
    """
    l = a[0] # low
    m = a[len(a)/2] # midpoint 
    h = a[-1] # high
    summed = 0 # sum over the values
    lsum = -sys.maxint  # left sum. we set this as low as we can
    for i in range(m, l-1, -1):
         summed += a[i]
         if summed > lsum:
             lsum = summed
             ml = i # max-left 
    rsum = -sys.maxint
    summed = 0
    for i in range(m+1, h+1):
        summed += a[i]
        if sum > rsum:
            rsum = sum
            mr = i # max-right
    return (ml, mr, lsum + rsum)
