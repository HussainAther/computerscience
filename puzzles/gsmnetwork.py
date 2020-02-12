# python3
import itertools

"""
GSM network is a type of infrastructure used for communication via mobile phones. It includes
transmitting towers scattered around the area which operate in different frequencies. 

Typically there is one tower in the center of each hexagon called “cell” on the grid above — hence the name “cell phone”.
A cell phone looks for towers in the neighborhood and decides which one to use based on strength
of signal and other properties. For a phone to distinguish among a few closest towers, the frequencies
of the neighboring towers must be different. You are working on a plan of GSM network for mobile,
and you have a restriction that you’ve only got 3 different frequencies from the government which you
can use in your towers. You know which pairs of the towers are neighbors, and for all such pairs the
towers in the pair must use different frequencies. You need to determine whether it is possible to assign
frequencies to towers and satisfy these restrictions.
This is equivalent to a classical graph coloring problem: in other words, you are given a graph, and
you need to color its vertices into 3 different colors, so that any two vertices connected by an edge
need to be of different colors. Colors correspond to frequencies, vertices correspond to cells, and edges
connect neighboring cells. Graph coloring is an NP-complete problem, so we don’t currently know an
efficient solution to it, and you need to reduce it to an instance of SAT problem which, although it is
NP-complete, can often be solved efficiently in practice using special programs called SAT-solvers.
"""

n, m = map(int, input().split())
edges = [ list(map(int, input().split())) for i in range(m) ]

says = []
colors = range(1, 4)

def vari(i, k):
    return 3*(i-1) + k

def eoo(i):
    lits = [vari(i, k) for k in colors]
    says.append([l for l in lits])

    for pair in itertools.combinations(lits, 2):
        says.append([-l for l in pair])

def adj(i, j):
    for k in colors:
        says.append([-vari(i, k), -vari(j, k)])

for i in range(1, n+1):
    eoo(i)

for i, j in edges:
    adj(i, j)

print(len(says), n*3)
for c in says:
    c.append(0)
    print(" ".join(map(str, c)))
