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
        s = line.split(' ')
        if len(s) == 4:
            u, v, c, f = [int(e) for e in s]
        else:
            u, v, c = [int(e) for e in s]
            f = 0
        if u not in N:
            N[u] = {}
        if v not in N:
            N[v] = {}
        N[u][v] = {'cap': c, 'flow': f}
    return N
