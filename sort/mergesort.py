"""
Implement the Merge sort algorithm for a neato listo in Python.
"""

def mergeSort(listo):
    """
    Repeatedly call this until we get individual elements into their own lists.
    """
    if len(listo)>1:
        mid = len(listo)//2
        l = listo[:mid]
        r = listo[mid:]
        
        # recursive mergeSort
        mergeSort(l)
        mergeSort(r)

        i=0
        j=0
        k=0
        while i < len(l) and j < len(r):
            if l[i] < r[j]:
                listo[k]=l[i]
                i=i+1
            else:
                listo[k]=r[j]
                j=j+1
            k=k+1

        while i < len(l):
            listo[k]=l[i]
            i=i+1
            k=k+1

        while j < len(r):
            listo[k]=r[j]
            j=j+1
            k=k+1

lista = [1,2,3,2]
mergeSort(lista)
print(lista)
