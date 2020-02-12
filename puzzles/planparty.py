#python3

import sys
import threading

"""
You’re planning a company party. You’d like to invite the coolest people, and you’ve assigned each
one of them a fun factor — the more the fun factor, the cooler is the person. You want to maximize the
total fun factor (sum of the fun factors of all the invited people). However, you can’t invite everyone,
because if the direct boss of some invited person is also invited, it will be awkward. Find out what is
the maximum possible total fun factor.
"""

sys.setrecursionlimit(10**6) 
threading.stack_size(2**26)  
threading.Thread(target=main).start()

class Ver:
    def __init__(self, weight):
        self.children = []
        self.weight = weight

def rt():
    size = int(input())
    tree = [Ver(weight) for weight in map(int, input().split())]
    for i in range(1, size):
        a, b = list(map(int, input().split()))
        tree[a - 1].children.append(b - 1)
        tree[b - 1].children.append(a - 1)
    return tree

def dfs(tree, vertex, parent, D):
    if -1 == D[vertex]:
        if 1 == len(tree[vertex].children) and 0 != vertex:
            D[vertex] = tree[vertex].weight
        else:
            m1 = tree[vertex].weight
            for u in tree[vertex].children:
                if u != parent:
                    for w in tree[u].children:
                        if w != vertex:
                            m1 += dfs(tree, w, u, D)
            m0 = 0
            for u in tree[vertex].children:
                if u != parent:
                    m0 += dfs(tree, u, vertex, D)
            D[vertex] = max(m1, m0)
    return D[vertex]

def mw(tree):
    size = len(tree)
    if size == 0:
        return 0
    D = [-1] * size
    d = dfs(tree, 0, -1, D)
    return d

if __name__ == "__main__":
    tree = rt()
    weight = mw(tree)
    print(weight)

