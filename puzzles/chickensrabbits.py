"""
Write a program to solve a classic ancient Chinese puzzle: 
We count 35 heads and 94 legs among the chickens and rabbits in a farm. 
How many rabbits and how many chickens do we have?
"""

h = 35
l = 94

for i in range(h+1):
    j = h - i
    if 2*i + 4*j == l:
        print(i)
        print(j)
