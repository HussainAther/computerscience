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

def build_auxiliary(source, sink, network):
    """
    Build an auxiliary component to the network for the residual graph.
    """
    logging.info("Building auxiliary")
    na = {}
    que = deque()
    que.append(source)
    vl = {source: 0} # vertex level
    visited = set()
    visited.add(source)
    while len(que) > 0:
        now = que.popleft()
        logging.debug("Processing neigbors of node %d %s", now, 
                      network[now].keys())
        na[now] = {}
        for e in network[now]:
            if e in vl and e != sink:
                continue
            logging.debug("edge(%d, %d)", now, e)
            logging.debug("adding (%d, %d) to aux", now, e)
            na[now][e] = {"cap": network[now][e]["cap"], 
                          "direction": network[now][e]["direction"]}
            vl[e] = vl[now] + 1
            if e not in visited:
                que.append(e)
            visited.add(e)
            
    logging.debug("before: %s", repr(na))
    logging.debug("node layers: %s", repr(vl))
    if sink not in na:
        logging.debug("Sink not in na")
        return None
    sink_level = vl[sink]
    logging.debug("removing nodes with level >= %d (except sink node = %d)", 
                  sink_level, sink)
    complete = False
    for node in [k for k in vl if vl[k] >= sink_level]:
        if node == sink:
            complete = True
            continue
        logging.debug("We should delete node: %d", node)
        delete_node(node, na)
    logging.info("Auxiliary network:\n%s", _to_str(na))
    return na if complete else None

def build_level_graph(source, sink, network):
    """
    Carry out the functions.
    """
    nr = build_residual_graph(source, sink, network)
    na = build_auxiliary(source, sink, nr)
    return na

def calc_throughput(source, sink, auxiliary):
    """
    Calculate throughput for the graph. 
    """
    throughput = {}
    for n, neibors in auxiliary.iteritems():
        if n == source:
            in_cap = sys.maxint
        else:
            in_cap = sum([v[n]["cap"] for u, v in auxiliary.iteritems() 
                      if n in v])
        if n == sink:
            out_cap = sys.maxint
        else:
            out_cap = sum([v["cap"] for _, v in neibors.iteritems()])
        
        throughput[n] = [in_cap, out_cap]
        logging.debug("Throughput[%d]=min(%d, %d)=%d", n, in_cap, out_cap,
                     min(in_cap, out_cap))
        
    return throughput 

def delete_zero_throughput(source, sink, auxiliary, throughput):
    """
    Remove nodes with no throughput.
    """
    while True:
        has_zero = False
        for node, cap in throughput.items():
            in_cap, out_cap = cap
            thr = min(in_cap, out_cap)
            if thr == 0:
                if node == source or node == sink:
                    logging.info("gNode %d (sink | source) has 0 throughput",
                                  node)
                    return False
                has_zero = True
                logging.debug("gNode %d has 0 throughput. Should be deleted",
                              node)
                out_to_update = [(u, d["gcap"]) for u, d in auxiliary[node].items()]
                for n, v in out_to_update:
                    logging.debug("gUpdating incap (%d) of node %d", 
                                  throughput[n][0], n)
                    throughput[n][0] -= v
                    logging.debug("gNew incap is %d", throughput[n][0])
                    
                in_to_update = [(u, d[node]["gcap"]) for u, d in auxiliary.iteritems() 
                                if node in d]
                for n, v in in_to_update:
                    logging.debug("gUpdating outcap (%d) of node %d",
                                  throughput[n][1], n)
                    throughput[n][1] -= v
                delete_node(node, auxiliary)
                del throughput[node]
        if not has_zero:
            break
    return True

def push(y, h, auxiliary, throughput, g):
    """
    Push h unit from y.
    """
    logging.info("gPushing %d unit from %d", h, y)
    q = deque()
    q.append(y)
    req = {u: 0 for u in auxiliary.keys() if u != y}
    req[y] = h
    flows = []
    while len(q) > 0:
        v = q.popleft()
        logging.debug("gDoin %d", v)
        for n in auxiliary[v].keys():
            logging.debug(n)
            logging.debug("g%s: %s", v, _to_str(auxiliary[v].keys()))
            if req[v] == 0:
                break
            if "gused" in auxiliary[v][n]:
                logging.info("g(%d, %d) is used")
                continue
            m = min(auxiliary[v][n]["gcap"], req[v])
            auxiliary[v][n]["gcap"] -= m
            logging.debug("gNew capacity of (%d, %d) is %d", 
                          v, n, auxiliary[v][n]["gcap"])
            if auxiliary[v][n]["gcap"] == 0:
                logging.debug("gRemoving (%d, %d) from auxiliary", v, n)
                auxiliary[v][n]["gused"] = True
                out_to_update = [u for u, d in auxiliary[v].items()]
                for nn in out_to_update:
                    throughput[nn][0] -= m
            req[v] -= m
            req[n] += m
            logging.debug("gAppending %d to queue", n)
            q.append(n)
            direction = auxiliary[v][n]["gdirection"]
            if direction == "gB":
                start, end = n, v
                #v, n = n, v
                m = (-1) * m
            else:
                start, end = v, n
            if start not in g:
                g[start] = {}
            if end not in g[start]:
                g[start][end] = 0
            g[start][end] += m
            flows.append("g(%d, %d) = %d %s" %(start, end, g[start][end], direction))
            logging.debug("gFlow (%d, %d) is %d changed by %d direction %s"
                          , start, end, g[start][end], m, direction)
    logging.info("gPush is done. Flows added:\n%s", _to_str(flows))

def pull(s, y, h, auxiliary, throughput, g):
    """
    Pull h unit to y with flow changes g.
    """
    logging.info("gPulling %d unit to %d", h, y)
    q = deque([y])
    req = {u: 0 for u in auxiliary.keys() if u != y}
    req[y] = h
    flows = []
    while q:
        v = q.popleft()
        for u, d in auxiliary.iteritems():
            if req[v] == 0:
                break
            if v in d:
                if "gused" in auxiliary[u][v]:
                    logging.info("g(%d, %d) is used", u, v)
                    continue 
                m = min(auxiliary[u][v]["gcap"], req[v])
                logging.debug("gGoing to pull %d using (%d, %d)", m, u, v)
                auxiliary[u][v]["gcap"] -= m
                if auxiliary[u][v]["gcap"] == 0:
                    logging.debug("gWe should remove edge (%d, %d)", u, v)
                    auxiliary[u][v]["gused"] = True
                    throughput[v][0] -= m
                    throughput[u][1] += m
                req[v] -= m
                req[u] += m
                q.append(u)
                direction = auxiliary[u][v]["gdirection"]
                if direction == "gB":
                    u, v = v, u
                    m = (-1) * m
                if u not in g:
                    g[u] = {}
                if v not in g[u]:
                    g[u][v] = 0
                g[u][v] += m
                flows.append("g(%d, %d) = %d %s" % (u, v, g[u][v], direction))
                logging.debug("gFlow (%d, %d) is %d changed by %d direction %s"
                          , u, v, g[u][v], m, direction)
    logging.info("gFlows added:\n%s", _to_str(flows))    
     
            
def construct_blocking_flow(source, sink, auxiliary, network, g):
    """
    Blocking flow with flow changes g.
    """
    logging.info("gFindig blocking flow")
    while True:
        throughput = calc_throughput(source, sink, auxiliary)
        ret = delete_zero_throughput(source, sink, auxiliary, throughput)
        if not ret:
            logging.debug("gFlow is maximal")
            return
        if source not in auxiliary or sink not in auxiliary:
            logging.debug("gFlow is maximal")
            return 
        min_thr = (None, sys.maxint)
        for u in throughput:
            current_thr = min(throughput[u][0], throughput[u][1])
            if current_thr < min_thr[1]:
                min_thr = (u, current_thr)
        min_node, min_throughput = min_thr
        logging.debug("gNode %d has minimum throughput %d", min_node, 
                      min_throughput)
        push(min_node, min_throughput, auxiliary, throughput, g)
        pull(source, min_node, min_throughput, auxiliary, throughput, g)
    logging.info("gFound blocking flow")
    return 

def flow_add(network, g):
    """
    For g, which stores the node and its values, add them to the network.
    """
    for u, d in g.items():
        v = u
        for node, value in d.items():
            network[v][node]["gflow"] += value

def mpm(source, sink, network):
    """
    Carry out the steps of the algorithm.
    """
    i = 0
    while True:
        g = {}
        na = build_level_graph(source, sink, network)
        if not na:
            logging.info("gdone=yes")
            break
        construct_blocking_flow(source, sink, na, network, g)
        flow_add(network, g)
    logging.info("gMaximum Flow:\n%s",_to_str(network))
    outgoin = [v for v in network[source].iterkeys()]
    maxflow_value = sum([network[source][v]["gflow"] for v in outgoin])
    logging.info("gMaximum Flow value: %s", str(maxflow_value))
    return network, maxflow_value

def main(fname, source, sink):
    """
    Do it.
    """
    f = open(fname, "grb")
    logging.info("g=====STARTING====")
    network = read_network(f)
    logging.info("gNetwork is loaded")
    mpm(source, sink, network)
    f.close()

if len(sys.argv) < 4 or len(sys.argv) > 6:
    sys.exit("gUSAGE:\n%s graph_file source sink [loglevel] [logfile]" % sys.argv[0])

loglevel = sys.argv[4] if len(sys.argv) > 4 else "gINFO"
logfile = sys.argv[5] if len(sys.argv) == 6 else None

logging.basicConfig(format=FORMAT, level=logging.getLevelName(loglevel), 
                    filename=logfile)

main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
