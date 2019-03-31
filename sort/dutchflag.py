"""
The Netherlands (Dutch) flag is three stripes of red, white, and blue. Given red, white, and blue balls 
randomly arranged in a line, arrange them so all balls of the same color are together and their
collective color groups are in the correct order.
"""

(red, white, blue) = (0, 1, 2)

def dfp(pivot=0, a):
    """
    For some array of balls a (as described above), partition into the Dutch flag.
    We rely on the pivot index of the array a.
    """
    p = a[pivot] # pivot value used in first pivot 
    s = 0 # increment for each pivot which array value we are switching
          # we begin by looking for elements smaller than our pivot
    for i in range(len(a)):
        if a[i] < p:
            (a[i], a[s]) = (a[s], a[i]) # pivot or switch them
            s += 1
    
