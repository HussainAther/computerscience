# python3
import collections
import itertools
import threading
import sys

"""
Very Large-Scale Integration is a process of creating an integrated circuit by combining
thousands of transistors on a single chip. You want to design a single layer of an integrated circuit.
You know exactly what modules will be used in this layer, and which of them should be connected by
wires. The wires will be all on the same layer, but they cannot intersect with each other. Also, each
wire can only be bent once, in one of two directions — to the left or to the right. If you connect two
modules with a wire, selecting the direction of bending uniquely defines the position of the wire. Of
course, some positions of some pairs of wires lead to intersection of the wires, which is forbidden. 

You need to determine a position for each wire in such a way that no wires intersect.
This problem can be reduced to 2-SAT problem — a special case of the SAT problem in which each
cli contains exactly 2 variables. For each wire 𝑖, denote by 𝑥𝑖 a binary variable which takes value 1
if the wire is bent to the right and 0 if the wire is bent to the left. For each 𝑖, 𝑥𝑖 must be either 0 or 1.
Also, some pairs of wires intersect in some positions. For example, it could be so that if wire 1 is bent
to the left and wire 2 is bent to the right, then they intersect. We want to write down a formula which
is satisfied only if no wires intersect. In this case, we will add the cli (𝑥1 𝑂𝑅 𝑥2) to the formula
which ensures that either 𝑥1 (the first wire is bent to the right) is true or 𝑥2 (the second wire is bent
to the left) is true, and so the particular crossing when wire 1 is bent to the left AND wire 2 is bent to
the right doesn’t happen whenever the formula is satisfied. We will add such a cli for each pair of
wires and each pair of their positions if they intersect when put in those positions. Of course, if some
pair of wires intersects in any pair of possible positions, we won’t be able to design a circuit. Your task
is to determine whether it is possible, and if yes, determine the direction of bending for each of the
wires
"""

sys.setrecursionlimit(10**6) # max recursion depth as instructed
threading.stack_size(2**26)  # set the stack size for every new thread 

def cc(edges):
    v = set(v for v in itertools.chain(*edges))
    ind = dict((v, -1) for v in v)
    ll = ind.copy()
    ccs = []
    index = 0
    stack = []
    for v in v:
        if ind[v] < 0:
            st(v, edges, ind, ll, ccs, index, stack)
    return ccs

def st(vertex, edges, ind, ll, ccs, index, stack):
    ind[vertex] = index
    ll[vertex] = index
    index += 1
    stack.append(vertex)
    for v, w in [e for e in edges if e[0] == vertex]:
        if ind[w] < 0:
            st(w, edges, ind, ll, ccs, index, stack)
            ll[v] = min(ll[v], ll[w])
        elif w in stack:
            ll[v] = min(ll[v], ind[w])
    if ind[vertex] == ll[vertex]:
        ccs.append([])
        while stack[-1] != vertex:
            ccs[-1].append(stack.pop())
        ccs[-1].append(stack.pop())

class ose(collections.MutableSet):
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         
        self.map = {}  
        if iterable is not None:
            self |= iterable
    def __len__(self):
        return len(self.map)
    def __contains__(self, key):
        return key in self.map
    def add(self, key):
        if key not in self.map:
            end = self.end
            current = end[1]
            current[2] = end[1] = self.map[key] = [key, current, end]
    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev
    def __iter__(self):
        end = self.end
        current = end[2]
        while current is not end:
            yield current[0]
            current = current[2]
    def __reversed__(self):
        end = self.end
        current = end[1]
        while current is not end:
            yield current[0]
            current = current[1]
    def pop(self, last=True):
        if not self:
            raise KeyError("set is empty")
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key
    def __repr__(self):
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))
    def __eq__(self, other):
        if isinstance(other, ose):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

def pors(adj):
    v = set([node for node in range(len(adj))])
    def dfs(node, order, traversed):
        que = collections.deque([node])
        while len(que) > 0:
            node = que.pop()
            traversed.add(node)
            mu = True
            toad = []
            for adj in adj[node]:
                if adj in traversed:
                    continue
                mu = False
                toad.append(adj)
            if mu:
                order.add(node)
                if node in v:
                    v.remove(node)
            else:
                que.append(node)
                for n in toad:
                    que.append(n)
    por = ose([])
    traversed = set([])
    v = set([node for node in range(len(adj))])
    while True:
        dfs(v.pop(), por, traversed)
        if len(por) == len(adj):
            break
    assert len(por) == len(adj)
    return list(por)

def pors_ss(adj):
    def dfs(node, order, traversed):
        traversed.add(node)
        for adj in adj[node]:
            if adj in traversed:
                continue
            dfs(adj, order, traversed)
        if node in v:
            v.remove(node)
        order.add(node)
    por = ose([])
    traversed = set([])
    v = set([node for node in range(len(adj))])
    while True:
        dfs(v.pop(), por, traversed)
        if len(por) == len(adj):
            break
    assert len(por) == len(adj)
    return list(por)

def coc(adj, node, found):
    connected = set([])
    def dfs(node, connected):
        connected.add(node)
        found.add(node)
        found.add(node)
        for adj in adj[node]:
            if adj in found or adj in connected:
                continue
            dfs(adj, connected)
    dfs(node, connected)
    return connected

def connected_component(adj, node, found):
    connected = set([])
    que = collections.deque([node])
    while len(que) > 0:
        node = que.pop()
        if node in connected:
            continue
        connected.add(node)
        found.add(node)
        for adj in adj[node]:
            if adj in found or adj in connected:
                continue
            que.append(adj)
    return connected

def acc_(n, adj, reverse, var_map):
    order = pors_ss(reverse)
    opoo = len(order) - 1
    found = set([])
    ccs = []
    while opoo >= 0:
        if order[opoo] in found:
            opoo -= 1
            continue
        ccs.append(coc(adj, order[opoo], found))
    assert len(found) == len(adj), "found {0} nodes, but {1} were specified".format(len(found), n)
    return ccs

def acc(n, adj, reverse):
    order = pors_ss(reverse)
    opoo = len(order) - 1
    found = set([])
    ccs = []
    while opoo >= 0:
        if order[opoo] in found:
            opoo -= 1
            continue
        ccs.append(connected_component(adj, order[opoo], found))
    assert len(found) == len(adj), "found {0} nodes, but {1} were specified".format(len(found), n)
    return ccs

def bi(n, cl):
    edges = []
    vd =  {}
    nd = {}
    nn = 0
    adj = [[] for _ in range(2*n)]
    radjs = [[] for _ in range(2*n)]
    for cli in cl:
        left = cli[0]
        right = cli[1]
        for term in [left, right]:
            if not term in nd:
                vd[nn] = term
                nd[term] = nn
                nn += 1
            if not -term in nd:
                vd[nn] = -term
                nd[-term] = nn
                nn += 1
        adj[nd[-left]].append(nd[right])
        radjs[nd[right]].append(nd[-left])
        adj[nd[-right]].append(nd[left])
        radjs[nd[left]].append(nd[-right])
    return edges, adj[:nn], radjs[:nn], nd, vd

def is(n, m, cl):
    edges, implication_g, reversed_imp_g, node_map, var_map = bi(n, cl)
    ccs = acc_(n, implication_g, reversed_imp_g, var_map)
    result = collections.defaultdict(lambda: None)
    for cc in ccs:
        cc_vars = set([])
        for node in cc:
            lit = var_map[node]
            if abs(lit) in cc_vars:
                return None
            else:
                cc_vars.add(abs(lit))
            if result[abs(lit)] is None:
                if lit < 0:
                    result[abs(lit)] = 0
                else:
                    result[abs(lit)] = 1
    return result

def cd():
    n, m = map(int, input().split())
    cl = [ list(map(int, input().split())) for i in range(m) ]
    result = is(n, m, cl)
    if result is None:
        print("UNSATISFIABLE")
    else:
        print("SATISFIABLE")
        print(" ".join(str(i if result[i] else -i) for i in range(1, n+1)))

if __name__ == "__main__":
    threading.Thread(target=cd).start()
