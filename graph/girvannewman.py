import sys
import csv
import networkx as nx

"""
Run using:

"python girvannewman.py cgraph.dat"

or replace "cgraph.dat" with whatever graph file you want.

The Girvan–Newman algorithm (named after Michelle Girvan and Mark Newman) is a hierarchical
method used to detect communities in complex systems. It constructs a graph by detecting communities
in the network.

The steps are generally:
1. The betweenness of all existing edges in the network is calculated first.
2. The edge with the highest betweenness is removed.
3. The betweenness of all edges affected by the removal is recalculated.
4. Steps 2 and 3 are repeated until no edges remain.
"""

def buildGraph(G, file_, delimiter_):
    """
    Build a graph from the cgraph.dat file or whatever input file is given.
    """
    reader = csv.reader(open(file_), delimiter=delimiter_)
    for line in reader:
        if len(line) > 2:  # for the format of cgraph.dat
            if float(line[2]) != 0.0: # if there are 3 edges
                G.add_edge(int(line[0]),int(line[1]),weight=float(line[2]))
        else: # 2 edges
            G.add_edge(int(line[0]),int(line[1]),weight=1.0)

def gnStep(g):
    """
    Perform the steps of the algorithm
    """
    init_ncomp = nx.number_connected_components(g) # get the number of components
    ncomp = init_ncomp # initialize with the initial number of components
    while ncomp <= init_ncomp: # loop through the steps as the algorithm dictates
        w = nx.edge_betweenness_centrality(g, weight="weight") # centrality of the edges
        themax = max(bw.values()) # max
        for k, v in bw.iteritems():
            if float(v) == themax: # if that edge is the maximum
                g.remove_edge(k[0], k[1]) # remove that edge
        ncomp = nx.number_connected_components(g) # recalcualte and perform the algorithm again

def gnModularity(g, deg, m):
    """
    Compute modularity of the part of the matrix
    """
    new = nx.adj_matrix(g) # get the adjacency matrix
    dego = Deg(new, g.nodes()) # degree of the matrix
    com = nx.connected_components(g) # get the connected components
    modo = 0 # modularity of the partition
    for c in com:
        ewc = 0 # edges in a community
        re = 0 # random edges
        for i in c:
            ewc += dego[i]
            re += deg[i]
        modo += (float(ewc) - float(re**2) / float(2*m)) # modularity of the matrix
    modo = modo/float(2*m) # update modularity
    return modo

def Deg(a, nodes):
    """
    Update the degrees of the matrix and return a dictionary of the
    updated nodes.
    """
    dicto = {} # that's spanish for dicto
    b = a.sum(axis=1)
    for i in range(len(nodes)):
        dicto[nodes[i]] = b[i, 0]
    return dicto

def gnRun(g, orig, m):
    """
    Run the Girvan-Newman algorithm.
    """
    best = 0 # best graph split
    while True: # iterate until we use all the number of edges
        gnStep(g)
        q = gnModularity(g, orig, m)
        if q > best: # if the modularity of this method of partitioning is better
            best = q
            bestcom = nx.connected_components(g) # best way of partitioning the graph
        if g.number_of_edges() == 0:
            break # break the loop when we get to this part of the algorithm
    if best > 0:
        print("Modularity (Q): %f " % best)
        print("Graph communities:", bestcom)
    else:
        print("Modularity (Q): %f" % best)

gr = sys.argv[1] # read input graph file
g = nx.graph() # initialize is using networkx
buildGraph(g, gr, ",") # build the graph

n = g.number_of_nodes()
a = nx.adj_matrix(g) # adjacency matrix
m = 0 # number of edges
for i in range(n):
    for j in range(n):
        m += a[i, j]
m = m / 2 # normalize
gnRun(g, Deg(a, g.nodes()), m) # run the algorithm after getting the weighted degree for each node
