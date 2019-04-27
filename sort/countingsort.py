"""
Counting sort is a sorting technique based on keys between a specific range.
It works by counting the number of objects having distinct key values (kind of hashing).
Then doing some arithmetic to calculate the position of each object in the output sequence.
"""

def countSort(a, exp):
    """
    Counting sort for an array a.
    """
    output = [0 for i in range(256)]
    c = [0 for i in range(256)]
    ans = ["" for _ in a]
    for i in a:
        c[ord(i)] += 1 # ord returns the unicode code point of the character in a string of length one.
    for i in range(256):
        c[i] += c[i-1]
    for i in range(len(a)):
        output[c[ord(a[i])]-1] = a[i]
        c[ord(a[i])] -= 1
    for i in range(len(a)):
        ans[i] = output[i]
    return ans

"""
Radix sort sorts data with integer keys by grouping keys by the individual
digits which share significant position and value.
"""

def radixSort(a):
    """
    Group keys by digits to sort.
    """
    max1 = max(a)
    exp = 1
    while max1/exp > 0:
        countSort(a, exp)
        exp *= 10
