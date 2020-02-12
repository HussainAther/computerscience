# python3

import itertools

"""
You’ve just had a huge party in your parents’ house, and they are returning tomorrow. You need
to not only clean the apartment, but leave no trace of the party. To do that, you need to clean all
the rooms in some order. After finishing a thorough cleaning of some room, you cannot return to it
anymore: you are afraid you’ll ruin everything accidentally and will have to start over. So, you need to
move from room to room, visit each room exactly once and clean it. You can only move from a room
to the neighboring rooms. You want to determine whether this is possible at all.
This can be reduced to a classic Hamiltonian Path problem: given a graph, determine whether there is
a route visiting each vertex exactly once. Rooms are vertices of the graph, and neighboring rooms are
connected by edges. Hamiltonian Path problem is NP-complete, so we don’t know an efficient algorithm
to solve it. You need to reduce it to SAT, so that it can be solved efficiently by a SAT-solver.
"""

n, m = map(int, input().split())
edges = [ list(map(int, input().split())) for i in range(m) ]

cla = []
posits = range(1, n+1)
adj = [[] for _ in range(n)]
for i, j in edges:
    adj[i-1].append(j-1)
    adj[j-1].append(i-1)

def varii(i, j):
    return n*i + j

def eoo(literals):
    cla.append([l for l in literals])
    for pair in itertools.combinations(literals, 2):
        cla.append([-l for l in pair])

for i in range(n):
    eoo([varii(i, j) for j in posits])

for j in posits:
    eoo([varii(i, j) for i in range(n)])

for j in posits[:-1]:
    for i, nodes in enumerate(adj):
        cla.append([-varii(i, j)] + [varii(n, j+1) for n in nodes])

print(len(cla), n*n)

for c in cla:
    c.append(0)
    print(" ".join(map(str, c)))
