"""
In the Tower of Hanoi (tower hanoi) puzzle, we have three rods and a number of disks of different sizes, which can 
slide onto any rod. The puzzle starts with the disks in a neat stack in ascending order of size on one rod, the 
smallest at the top. We must move the entire stack to another rod. Only one disk can be moved at a time. Each 
move consists of taking the upper disk from one of the stacks and placing it on top of another stack or 
on an empty rod. No larger disk may be placed on top of a smaller disk.
"""

result = []

def hanoi(n, i, m, f):
    """
    For the number of disks and the numbers representing the initial
    peg i, middle peg m, and final peg f (each in list form), solve the 
    Hanoi problem. This function moves n disks from initial to middle, 
    then from middle to final. 
    """
    if n > 0:
        hanoi(n-1, i, m, f) # perform the Hanoi problem with one fewer disk
        result.append(i.pop()) # append the final item from i to result
        move(n-1, m, f, i) # perform the Hanoi problem with one fewer disk from middle to initial
