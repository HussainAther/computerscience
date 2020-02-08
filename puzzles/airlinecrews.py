# python3
import copy
import queue

"""
The airline offers a bunch of flights and has a set of crews that can work on those flights. However,
the flights are starting in different cities and at different times, so only some of the crews are able to
work on a particular flight. You are given the pairs of flights and associated crews that can work on
those flights. You need to assign crews to as many flights as possible and output all the assignments.
"""

class solveit:
    def datain(self):
        n, m = map(int, input().split())
        am = [list(map(int, input().split())) for i in range(n)]
        return am
    def printout(self, m):
        line = [str(-1 if x == -1 else x + 1) for x in m]
        print(" ".join(line))
    def fm(self, am):
        n = len(am)
        m = len(am[0])
        m = [-1] * n
        br = [False] * m
        def bfs():
            vn = set()
            q = queue.Queue()
            q.put((1, None))
            vn.add((1, None))
            path = []
            parent = dict()
            while not q.empty():
                cn = q.get()
                if 1 == cn[0]: 
                    for i in range(n):
                        if -1 == m[i]:
                            vn.add((2, i))
                            parent[(2, i)] = (1, None)
                            q.put((2, i))
                elif 2 == cn[0]: 
                    i = cn[1]
                    for j in range(m):
                        if 1 == am[i][j] and j != m[i] and not (3, j) in vn:
                            vn.add((3, j))
                            parent[(3, j)] = cn
                            q.put((3, j))
                elif 3 == cn[0]:
                    j = cn[1]
                    if not br[j]:
                        a = cn
                        cn = (4, j)
                        while True:
                            path.insert(0, (a, cn))
                            if 1 == a[0]:
                                break
                            cn = a
                            a = parent[cn]
                        for e in path:
                            if 2 == e[0][0]:
                                m[e[0][1]] = e[1][1]
                            elif 3 == e[0][0] and 4 == e[1][0]:
                                br[e[1][1]] = True
                        return True 
                    else:
                        for i in range(n):
                            if j == m[i] and not (2, i) in vn:
                                vn.add((2, i))
                                parent[(2, i)] = cn
                                q.put((2, i))
            return False 

        while bfs():
            continue
        return m
    def solve(self):
        am = self.datain()
        m = self.fm(am)
        self.printout(m)

if __name__ == "__main__":
    max_matching = solveit()
    max_matching.solve()