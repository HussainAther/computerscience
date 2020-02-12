# python3
from itertools import combinations, permutations

"""
A school bus needs to start from the depot early in the morning, pick up all the children from their
homes in some order, get them all to school and return to the depot. You know the time it takes to
get from depot to each home, from each home to each other home, from each home to the school and
from the school to the depot. You want to define the order in which to visit children’s homes so as to
minimize the total time spent on the route.
This is an instance of a classical NP-complete problem called Traveling Salesman Problem. Given a
graph with weighted edges, you need to find the shortest cycle visiting each vertex exactly once. Vertices
correspond to homes, the school and the depot. Edges weights correspond to the time to get from one
vertex to another one. Some vertices may not be connected by an edge in the general case.
"""

inf = 10 ** 9

def datain():
    n, m = map(int, input().split())
    graph = [[inf] * n for _ in range(n)]
    for _ in range(m):
        u, v, weight = map(int, input().split())
        u -= 1
        v -= 1
        graph[u][v] = graph[v][u] = weight
    return graph

def printt( wop, path ):
    print(wop)
    if wop == -1:
        return
    print(" ".join(map(str, path)))

def opto_bf( graph ):
    n = len(graph)
    ban = inf
    bpa = []
    for p in permutations(range(n)):
        cs = 0
        for i in range(1, n):
            if graph[p[i - 1]][p[i]] == inf:
                break
            cs += graph[p[i - 1]][p[i]]
        else:
            if graph[p[-1]][p[0]] == inf:
                continue
            cs += graph[p[-1]][p[0]]
            if cs < ban:
                ban = cs
                bpa = list(p)
    if ban == inf:
        return (-1, [])
    return (ban, [x + 1 for x in bpa])

def opto( graph ):
    n = len(graph)
    C = [[inf for _ in range(n)] for __ in range(1 << n)]
    bk = [[(-1, -1) for _ in range(n)] for __ in range(1 << n)]
    C[1][0] = 0
    for size in range(1, n):
        for S in combinations(range(n), size):
            S = (0,) + S
            k = sum([1 << i for i in S])
            for i in S:
                if 0 != i:
                    for j in S:
                        if j != i:
                            curr = C[k ^ (1 << i)][j] + graph[i][j]
                            if curr < C[k][i]:
                                C[k][i] = curr
                                bk[k][i] = (k ^ (1 << i), j)
    bre, ci2 = min([(C[(1 << n) - 1][i] + graph[0][i], i) for i in range(n)])
    if bre >= inf:
        return (-1, [])
    bP = []
    ci1 = (1 << n) - 1
    while -1 != ci1:
        bP.insert(0, ci2 + 1)
        ci1, ci2 = bk[ci1][ci2]
    return (bre, bP)

if __name__ == "__main__":
    printt(*opto(datain()))
