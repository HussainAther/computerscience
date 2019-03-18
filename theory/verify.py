def Path(edges):
    return list(edges) + list(edges[0])

def isHamiltonCycle(cycle):
    for i in cycle:
        if cycle.count(i) != 1:
            return "no"
    return "yes"

def cycleLength(cycle):
    length = 0
    for i in range(len(cycle)):
        length += cycle[i+1] - cycle[i]
    return length

"""
Complexity classes in computer science have varying times in finding solutions. Verifying that
solutions exist exist in complexity classes.
"""

def verifyFactor(I, S, H):
    """
    Checks solutinos to computational problems. It takes three strings: I instance of the problem,
    S the proposed soultion, and H a hint.
    """
    if S == "no":
        return "unsure"
    M = int(I)
    m = int(S)
    if m >= 2 and m < M and M % m == 0:
        # m is a nontrivial factor of M
        return "correct"
    else:
        # m is nontrivial
        return "unsure"

"""
In the next example uses the hint H to work in a reasonable amount of time. Decision variant of the Traveling Salesman, which originally takes
an undirected, weighted graph G and outputs a solution of the shortest Hamilton cycle of G (or "no" if none exists).
"""

def verifyTravelingSalesmanDecision(I, S, H):
    if S == "no":
        return "unsure"
    # extract G, L, and I and convert to correct data types
    (G, L) = I.split(";")

    # split the hint string into a list of verticies, which will
    # form a Hamilton cycle of length at most L, if the hint is correct
    cycle = Path(H.split(","))

    # verify the hint is a Hamilton cycle, and has length at most L
    if G.isHamiltonCycle(cycle) and G.cycleLength(cycle) <= L: # A Hamilton cycle visits each node only once, except the starting node, which it visits twice
        return "correct"
    else:
        return "unsure"

def verifyCheckMultiply(I, S, H):
    (M1, M2, K) = [int() for x in I.split()]
    if M1*M2 == K and S == "yes":
        return "correct"
    elif M1*M2 != K and S == "no":
        return "correct"
    return "unsure"

def verifyAdd(I, S, H):
    if len(S)>2*len(I) or len(H)>0:
        return "unsure"
    (M1, M2) = [int(s) for x in I.split()]
    S = int(S)
    total = M1
    for i in range(M2):
        total += 1
    if total == S:
        return "correct"
    return "unsure"
