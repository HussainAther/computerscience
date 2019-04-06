"""
Selection sort uses the bubble sort but only one exchange for every pass through the list.
"""

def selectionSort(listo):
   for f in range(len(listo)-1, 0, -1):
       p = 0
       for l in range(1, f+1):
           if listo[l] > listo[p]:
               p = l

       t = listo[f]
       listo[f] = listo[p]
       listo[p] = t
