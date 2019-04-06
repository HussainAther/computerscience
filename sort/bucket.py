"""
Bucket sort uses the input drawn from a uniform 
distribution that has an average-case running time of O(n).
"""

def ins(a):
    """
    Insertion sort for array a.
    """
    for i in range(1, len(a)):
        c = a[i]
        p = i
        while p > 0 and a[p-1] > c:
            a[p] = a[p-1]
            p -= 1
        a[p] = c
 

def bucket(a):
    """
    Using Bucket sort on an array a
    """
    b = [[]]*len(a) # output
    for i in range(len(a)):
        b[i].append(a[i])
    for i in range(len(a)):
        ins(b[i])
    r = []
    for i in b:
        for j in i:
            r.append(j)
    return r 
  
