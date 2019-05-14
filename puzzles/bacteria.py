"""
There are n bacteria and 1 virus in a Petri dish. Within the first minute, the virus kills 
one bacterium and produces another copy of itself, and all of the remaining bacteria reproduce, 
making 2 viruses and 2 · (n − 1) bacteria. In the second minute, each of the viruses kills 
a bacterium and produces a new copy of itself (resulting in 4 viruses and 2(2(n − 1) − 2) = 4n − 8 
bacteria; again, the remaining bacteria reproduce. This process continues every minute. Will 
the viruses eventually kill all the bacteria? If so, design an algorithm that computes how 
many steps it will take. How does the running time of your algorithm depend on n?
"""

def bacteria(b, v):
    """
    For b bacteria and v viruses at the beginning, we can use recurrence relations of b(t)
    and v(t) (as they depend on time, to determine how long it would take for the viruses
    to kill all the bacteria.
    """
