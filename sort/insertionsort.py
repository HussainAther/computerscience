"""
Maintain a sorted sublist in the lower values of the list.
Insert each new item into it.
"""

def i(listo):
    """
    Insert sort inserts each item by order. 
    """
    for x in range(1,len(listo)):
        c = listo[x]
        p = x
        while p > 0 and listo[p-1] > c:
            listo[p] = listo[p-1]
            p = p-1
        listo[p] = c
    return listo
