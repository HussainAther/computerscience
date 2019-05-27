import sys
import logging
import pprint
import cStringIO as StringIO

from collections import deque

"""
Malhotra, Pramodh Kumar, and Maheshwari (MPM) algorithm.
The algorithm has time complexity of (n^3). The algorithm operates in stages.
We construct the network N(f) [residual network] and from it we find the
auxiliary[layered] network AN(f). Then we find a maximal flow g in AN(f) and 
update flows in the main network
"""

FORMAT = "%(asctime)-15s %(levelname)s - %(message)s"

def _to_str(network):
    """
    Return the network in string format.
    """
    out = StringIO.StringIO()
    pp = pprint.PrettyPrinter(indent=1, stream=out)
    pp.pprint(network)
    to_return = out.getvalue()
    out.close()
    return to_return
    

def read_network(f=sys.stdin):
    """
    Read the current netwrok as it is.
    """
    N = {}
    lines = f.readlines()
    for line in lines:
        s = line.split(" ")
        if len(s) == 4:
            u, v, c, f = [int(e) for e in s]
        else:
            u, v, c = [int(e) for e in s]
            f = 0
        if u not in N:
            N[u] = {}
        if v not in N:
            N[v] = {}
        N[u][v] = {"cap": c, "flow": f}
    return N

def delete_node(node, network):
    """
    Delete a node in the network.
    """
    for u, v in network.items():
        if node in v:
            logging.debug("Deleting edge: (%d, %d)", u, node)
            del v[node]
        if node in network:
            logging.debug("Removing node %d from network", node)
            del network[node]

def build_residual_graph(source, sink, network):
    """
    Build the graph using the source, sink, and network.
    """
    logging.debug("Building residual graph")
    nr = {}
    que = deque()
    que.append(source)
    visited = set()
    visited.add(source)
    while len(que) > 0:
        now = que.popleft()
        logging.debug("Processing neigbors of node %d", now)
        for e in network[now]:
            logging.debug("edge(%d, %d)", now, e)
            r = network[now][e]["cap"] - network[now][e]["flow"]
            logging.debug("residual cap is %d", r)
            if now not in nr:
                nr[now] = {}
            if e not in nr:
                nr[e] = {}
            if r > 0:
                nr[now][e] = {"cap": r ,"direction": "F"}
                logging.debug("adding (%d, %d) with cap = %d to na", now, e, r) 
            if network[now][e]["flow"] > 0:
                nr[e][now] = {"cap": network[now][e]["flow"], "direction": "B"}
                logging.debug("adding (%d, %d) with cap = %d to na", e, now,
                              network[now][e]["flow"])
            if e not in visited:
                que.append(e)
            visited.add(e)
    logging.info("Residual network:\n%s", _to_str(nr))
    return nr
 
