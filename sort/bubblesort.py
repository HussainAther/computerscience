"""
Bubble sort makes passes through a list and compares adjacent items
and exchanges them if they're not in order. Each item forms a bubble around the location
where it belongs.
"""
def b(listo):
    for p in range(len(listo)-1,0,-1):
        for i in range(p):
            if listo[i]>listo[i+1]:
                t = listo[i]
                listo[i] = listo[i+1]
                listo[i+1] = t
