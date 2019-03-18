"""
Quick sort algorithm on a list. Select a value (v) that pivot around it
to split the list around that point. Divide the list for subseqeunt calls.
"""


def quickSort(listo):
    h(listo,0,len(listo)-1)

def h(listo,first,last):
    if first<last:
        s = p(listo,first,last)
        h(listo,first,s-1)
        h(listo,s+1,last)

def p(listo,first,last):
    v = listo[first]

    l = first+1
    r = last

    d = False
    while not d:
        while l <= r and listo[l] <= v:
            l = l + 1

        while listo[r] >= v and r >= l:
            r = r -1
        if r < l:
            d = True
            else:
                temp = listo[l]
                listo[l] = listo[r]
                listo[r] = temp
    temp = listo[first]
    listo[first] = listo[r]
    listo[r] = temp
    return r
