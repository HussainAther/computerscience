# python3
import collections
import sys
import threading

from enum import Enum

"""
The new secretary at your Computer Science Department has prepared a schedule of exams for CS101: each student was assigned to his own exam date. However, it’s a disaster: not only some pairs of students known to be close 
friends may have been assigned the same date, but also NONE of the
students can actually come to the exam at the day they were assigned (there was a misunderstanding
between the secretary who asked to specify available dates and the students who understood they
needed to select the date at which they cannot come). There are three different dates the professors
are available for these exams, and these dates cannot be changed. The only thing that can be changed
is the assignment of students to the dates of exams. You know for sure that each student can’t come at
the currently scheduled date, but also each student definitely can come at any of the two other possible
dates. Also, you must make sure that no two known close friends are assigned to the same exam date.
You need to determine whether it is possible or not, and if yes, suggest a specific assignment of the
students to the dates.

This problem can be reduced to a graph problem called 3-recoloring. You are given a graph, and each
vertex is colored in one of the 3 possible colors. You need to assign another color to each vertex in
such a way that no two vertices connected by and edge are assigned the same color. Here, possible
colors correspond to the possible exam dates, vertices correspond to students, colors of the vertices
correspond to the assignment of students to the exam dates, and edges correspond to the pairs of close
friends.
"""

sys.setrecursionlimit(10 ** 6)  
threading.stack_size(2 ** 26) 

class Ordered_Sets(collections.MutableSet):
    def __init__( self, iterable=None ):
        self.end = end = []
        end += [None, end, end]  
        self.map = {}  # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable
    def __len__( self ):
        return len(self.map)
    def __contains__( self, key ):
        return key in self.map
    def add( self, key ):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]
    def discard( self, key ):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev
    def __iter__( self ):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]
    def __reversed__( self ):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]
    def pop( self, last=True ):
        if not self:
            raise KeyError("set is empty")
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key
    def __repr__( self ):
        if not self:
            return "%s()" % (self.__class__.__name__,)

def poo( adjs ):
    def dfs( node, order, traversed ):
        traversed.add(node)
        for adj in adjs[node]:
            if adj in traversed:
                continue
            dfs(adj, order, traversed)
        if node in vertices:
            vertices.remove(node)
        order.add(node)
    post_order = Ordered_Sets([])
    traversed = set([])
    vertices = set([node for node in range(len(adjs))])
    while True:
        dfs(vertices.pop(), post_order, traversed)
        if len(post_order) == len(adjs):
            break
    assert len(post_order) == len(adjs)
    return list(post_order)

def cc( adjs, node, found ):
    connected = set([])
    def dfs( node, connected ):
        connected.add(node)
        found.add(node)
        for adj in adjs[node]:
            if adj in found or adj in connected:
                continue
            dfs(adj, connected)
    dfs(node, connected)
    return connected

def ancc( n, adjs, reverse ):
    order = poo(reverse)
    order_pointer = len(order) - 1
    found = set([])
    ccs = []
    while order_pointer >= 0:
        if order[order_pointer] in found:
            order_pointer -= 1
            continue
        ccs.append(cc(adjs, order[order_pointer], found))
    assert len(found) == len(adjs), "found {0} nodes, but {1} were specified".format(len(found), n)
    return ccs

class ig(object):
    vd = {}
    ndi = {}
    adjs = None
    rads = None
    def __init__( self, n, cla ):
        node_num = 0
        self.adjs = [[] for _ in range(2 * n)]
        self.rads = [[] for _ in range(2 * n)]
        for clause in cla:
            left = clause[0]
            right = clause[1]
            for term in [left, right]:
                if not term in self.ndi:
                    self.vd[node_num] = term
                    self.ndi[term] = node_num
                    node_num += 1
                if not -term in self.ndi:
                    self.vd[node_num] = -term
                    self.ndi[-term] = node_num
                    node_num += 1
            self.adjs[self.ndi[-left]].append(self.ndi[right])
            self.rads[self.ndi[right]].append(self.ndi[-left])
            self.adjs[self.ndi[-right]].append(self.ndi[left])
            self.rads[self.ndi[left]].append(self.ndi[-right])
        self.adjs = self.adjs[:node_num]
        self.rads = self.rads[:node_num]

class Color(Enum):
    R = 0
    G = 1
    B = 2

def gnco( var ):
    node = (var - 1) // 3
    c = var % 3
    if c == 0:
        return node, Color(2)
    if c == 2:
        return node, Color(1)
    if c == 1:
        return node, Color(0)

def gs2( n, edges, colors ):
    red = Color(0)
    green = Color(1)
    blue = Color(2)
    rgb = set([red, green, blue])
    cla = []
    for node_ in range(1, n + 1):
        node = node_ * 3 - 2
        c1 = Color[colors[node_ - 1]]
        others = rgb.difference(set([c1]))
        c2 = others.pop()
        c3 = others.pop()
        c1v = node + c1.value
        c2v = node + c2.value
        c3v = node + c3.value
        cla += [[c2v, c3v], [-c2v, -c3v], [-c1v, -c1v]]
    for edge in edges:
        left = edge[0] * 3 - 2
        right = edge[1] * 3 - 2
        cla += [[-left, -right], [-(left + 1), -(right + 1)], [-(left + 2), -(right + 2)]]
    return cla

def anc( n, edges, colors ):
    nv = n * 3
    cla = gs2(n, edges, colors[0])
    graph = ig(nv, cla)
    ccs = ancc(nv, graph.adjs, graph.rads)
    result = collections.defaultdict(lambda: None)
    for cc in ccs:
        ccv = set([])
        for node in cc:
            l = graph.vd[node]
            if abs(l) in ccv:
                return None
            else:
                ccv.add(abs(l))
            if result[abs(l)] is None:
                if l < 0:
                    result[abs(l)] = 0
                else:
                    result[abs(l)] = 1
    rc = []
    for key in sorted(result.keys()):
        if result[key] == 1:
            node, color = gnco(key)
            rc.append(color.name)
    return rc

n, m = map(int, input().split())
colors = input().split()
edges = []
for i in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))
new_colors = anc(n, edges, colors)
if new_colors is None:
    print("Impossible")
else:
    print("".join(new_colors))
