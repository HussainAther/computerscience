# python3
import copy
import numpy as np

from sys import stdin

bigggg = 1e8

"""
You want to optimize your diet: that is, make sure that your diet satisfies all the recommendations
of nutrition experts, but you also get maximum pleasure from your food and drinks. For each dish
and drink you know all the nutrition facts, cost of one item, and an estimation of how much you like
it. Your budget is limited, of course. The recommendations are of the form “total amount of calories
consumed each day should be at least 1000” or “the amount of water you drink in liters should be at
least twice the amount of food you eat in kilograms”, and so on. You optimize the total pleasure which
is the sum of pleasure you get from consuming each particular dish or drink, and that is proportional
to the amount amount𝑖 of that dish or drink consumed.

The budget restriction and the nutrition recommendations can be converted into a system of linear
inequalities like ∑︀𝑚 𝑖=1 cost𝑖· amount𝑖 ≤ Budget, amount𝑖 ≥ 1000 and amount𝑖 − 2 · amount𝑗 ≥ 0, where
amount𝑖 is the amount of 𝑖-th dish or drink consumed, cost𝑖 is the cost of one item of 𝑖-th dish or
drink, and 𝐵𝑢𝑑𝑔𝑒𝑡 is your total budget for the diet. Of course, you can only eat a non-negative amount
amount𝑖 of 𝑖-th item, so amount𝑖 ≥ 0. The goal to maximize total pleasure is reduced to the linear
objective ∑︀𝑚 𝑖=1 amount𝑖 · pleasure𝑖 → max where pleasure𝑖 is the pleasure you get after consuming one
unit of 𝑖-th dish or drink (some dishes like fish oil you don’t like at all, so pleasure𝑖 can be negative).
Combined, all this is a linear programming problem which you need to solve now.
"""

class Equation:
    def __init__( self, a, b ):
        self.a = a
        self.b = b

class Position:
    def __init__( self, column, row ):
        self.column = column
        self.row = row

def primee( a, ur, uc ):
    mm = len(a)
    pe = Position(0, 0)
    while ur[pe.row]:
        pe.row += 1
    while uc[pe.column]:
        pe.column += 1
    while 0 == a[pe.row][pe.column] or ur[pe.row]:
        pe.row += 1
        if pe.row > mm - 1:
            return False, None
    return True, pe

def SwapLines( a, b, ur, pe ):
    a[pe.column], a[pe.row] = a[pe.row], a[pe.column]
    b[pe.column], b[pe.row] = b[pe.row], b[pe.column]
    ur[pe.column], ur[pe.row] = ur[pe.row], ur[
        pe.column]
    pe.row = pe.column

def ProcessPivotElement( a, b, pe ):
    n = len(a)
    mm = len(a[pe.row])
    scale = a[pe.row][pe.column]
    for j in range(mm):
        a[pe.row][j] /= scale
    b[pe.row] /= scale
    for i in range(n):
        if i != pe.row:
            scale = a[i][pe.column]
            for j in range(pe.column, n):
                a[i][j] -= a[pe.row][j] * scale
            b[i] -= b[pe.row] * scale

def MarkPivotElementUsed( pe, ur, uc ):
    ur[pe.row] = True
    uc[pe.column] = True

def solveeq( equation ):
    a = equation.a
    b = equation.b
    size = len(a)

    uc = [False] * size
    ur = [False] * size
    for step in range(size):
        solved, pe = primee(a, ur, uc)
        if not solved:
            return False, None
        SwapLines(a, b, ur, pe)
        ProcessPivotElement(a, b, pe)
        MarkPivotElementUsed(pe, ur, uc)

    return True, b

def addeqs( n, mm, A, b, Big_number ):
    for i in range(mm):
        e = [0.0] * mm
        e[i] = -1.0
        A.append(e)
        b.append(0.0)
    A.append([1.0] * mm)
    b.append(Big_number)

def checkit( n, mm, A, b, c, result, lastEquation, ans, bestScore ):
    for r in result:
        if r < -1e-3:
            return False, ans, bestScore
    for i in range(n):
        r = 0.0
        for j in range(mm):
            r += A[i][j] * result[j]
        if r > b[i] + 1e-3:
            return False, ans, bestScore
    score = 0.0
    for j in range(mm):
        score += c[j] * result[j]
    if score <= bestScore:
        return False, ans, bestScore
    else:
        if lastEquation:
            return True, 1, score
        else:
            return True, 0, score

def solveit( n, mm, A, b, c, Big_number=bigggg ):
    addeqs(n, mm, A, b, Big_number)
    l = n + mm + 1
    ans = -1
    bestScore = -float('inf')
    bestResult = None
    for x in range(2 ** l):
        usedIndex = [i for i in range(l) if ((x / 2 ** i) % 2) // 1 == 1]
        if len(usedIndex) != mm:
            continue
        lastEquation = False
        if usedIndex[-1] == l - 1:
            lastEquation = True
        As = [A[i] for i in usedIndex]
        bs = [b[i] for i in usedIndex]
        solved, result = solveeq(copy.deepcopy(Equation(As, bs)))
        if solved:
            isAccepted, ans, bestScore = checkit(n, mm, A, b, c, result, lastEquation, ans, bestScore)
            if isAccepted:
                bestResult = result
    return [ans, bestResult]

def solve0( n, mm, A, b, c ):
    res = linprog(-np.array(c), A, b)
    if 3 == res.status:
        ans = 1
        x = None
    elif 0 == res.status:
        ans = 0
        x = list(res.x)
    else:
        ans = -1
        x = None
    return ans, x

n, mm = list(map(int, stdin.readline().split()))
A = []
for i in range(n):
    A += [list(map(int, stdin.readline().split()))]
b = list(map(int, stdin.readline().split()))
c = list(map(int, stdin.readline().split()))

anst, ansx = solveit(n, mm, A, b, c)

if anst == -1:
    print("No solution")
if anst == 0:
    print("Bounded solution")
    print(' '.join(list(map(lambda x: '%.18f' % x, ansx))))
if anst == 1:
    print("Infinity")