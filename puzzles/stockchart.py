# python3
import queue

"""
You’re in the middle of writing your newspaper’s end-of-year economics summary, and you’ve decided
that you want to show a number of charts to demonstrate how different stocks have performed over the
course of the last year. You’ve already decided that you want to show the price of 𝑛 different stocks,
all at the same 𝑘 points of the year.

A simple chart of one stock’s price would draw lines between the points (0, 𝑝𝑟𝑖𝑐𝑒0),(1, 𝑝𝑟𝑖𝑐𝑒1), . . . ,(𝑘 −
1, 𝑝𝑟𝑖𝑐𝑒𝑘−1), where 𝑝𝑟𝑖𝑐𝑒𝑖 is the price of the stock at the 𝑖-th point in time.

In order to save space, you have invented the concept of an overlaid chart. An overlaid chart is the
combination of one or more simple charts, and shows the prices of multiple stocks (simply drawing a
line for each one). In order to avoid confusion between the stocks shown in a chart, the lines in an
overlaid chart may not cross or touch.

Given a list of 𝑛 stocks’ prices at each of 𝑘 time points, determine the minimum number of overlaid
charts you need to show all of the stocks’ prices.
"""

class sc:
    def datain( self ):
        n, k = map(int, input().split())
        sd = [list(map(int, input().split())) for i in range(n)]
        return sd

    def printout( self, result ):
        print(result)

    def mc( self, sd ):
        n = len(sd)
        k = len(sd[0])
        adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if all([x < y for x, y in zip(sd[i], sd[j])]):
                    adj[i][j] = 1
        m = [-1] * n
        b = [False] * n

        def bfs():
            t = set()
            q = queue.Queue()
            q.put((1, None))
            t.add((1, None))
            path = []
            parent = {}
            while not q.empty():
                c = q.get()
                if 1 == c[0]: 
                    for i in range(n):
                        if -1 == m[i]:
                            t.add((2, i))
                            parent[(2, i)] = c
                            q.put((2, i))
                elif 2 == c[0]: 
                    i = c[1]
                    for j in range(n):
                        if 1 == adj[i][j] and j != m[i] and not (3, j) in t:
                            t.add((3, j))
                            parent[(3, j)] = c
                            q.put((3, j))
                elif 3 == c[0]:
                    j = c[1]
                    if not b[j]:
                        p = c
                        c = (4, j)
                        while True:
                            path.insert(0, (p, c))
                            if 1 == p[0]:
                                break
                            c = p
                            p = parent[c]
                        for e in path:
                            if 2 == e[0][0]:
                                m[e[0][1]] = e[1][1]
                            elif 3 == e[0][0] and 4 == e[1][0]:
                                b[e[1][1]] = True
                        return True  
                    else:
                        for i in range(n):
                            if j == m[i] and not (2, i) in t:
                                t.add((2, i))
                                parent[(2, i)] = c
                                q.put((2, i))
            return False 

        while bfs():
            continue
        return len([0 for i in m if -1 == i])

    def solve( self ):
        sd = self.datain()
        result = self.mc(sd)
        self.printout(result)

if __name__ == "__main__":
    sc = sc()
    sc.solve()