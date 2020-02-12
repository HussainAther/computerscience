# python3
import itertools

from sys import stdin

"""
The marketing department of your big company has many subdepartments which control advertising
on TV, radio, web search, contextual advertising, mobile advertising, etc. Each of them has prepared
their advertising campaign plan, and of course you don’t have enough budget to cover all of their
proposals. You don’t have enough time to go thoroughly through each subdepartment’s proposals and
cut them, because you need to set the budget for the next year tomorrow. You decide that you will
either approve or decline each of the proposals as a whole.
There is a bunch of constraints you face. For example, your total advertising budget is limited. Also,
you have some contracts with advertising agencies for some of the advertisement types that oblige
you to spend at least some fixed budget on that kind of advertising, or you’ll see huge penalties, so
you’d better spend it. Also, there are different company policies that can be of the form that you
spend at least 10% of your total advertising spend on mobile advertising to promote yourself in this
new channel, or that you spend at least $1M a month on TV advertisement, so that people always
remember your brand. All of these constraints can be rewritten as an Integer Linear Programming: for
each subdepartment 𝑖, denote by 𝑥𝑖 boolean variable that corresponds to whether you will accept or
decline the proposal of that subdepartment. Then each constraint can be written as a linear inequality.
"""

n, m = list(map(int, stdin.readline().split()))
a = []

for i in range(n):
    a += [list(map(int, stdin.readline().split()))]
b = list(map(int, stdin.readline().split()))

cloooo = []

for i, cent in enumerate(a):
    non0coeffs = [(j, cent[j]) for j in range(m) if 0 != cent[j]]
    l = len(non0coeffs)
    for x in range(2**l):
        cs = [non0coeffs[j] for j in range(l) if 1 == ((x/2**j)%2)//1]
        csu = 0
        for coeff in cs:
            csu += coeff[1]
        if csu > b[i]:
            cloooo.append([-(coeff[0]+1) for coeff in cs] + [coeff[0]+1 for coeff in non0coeffs if not coeff in cs])

if 0 == len(cloooo):
    cloooo.append([1, -1])
    m = 1

print(len(cloooo), m)

for cc in cloooo:
    cc.append(0)
    print(" ".join(map(str, cc)))
