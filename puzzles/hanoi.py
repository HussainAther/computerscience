"""
In the Tower of Hanoi (tower hanoi) puzzle, we have three rods and a number of disks of different sizes, which can 
slide onto any rod. The puzzle starts with the disks in a neat stack in ascending order of size on one rod, the 
smallest at the top. We must move the entire stack to another rod. Only one disk can be moved at a time. Each 
move consists of taking the upper disk from one of the stacks and placing it on top of another stack or 
on an empty rod. No larger disk may be placed on top of a smaller disk.
"""

def hanoi(n, i, m, f):
    """
    For the number of disks and the numbers representing the initial
    peg i, middle peg m, and final peg f, solve the Hanoi problem. 
    """
