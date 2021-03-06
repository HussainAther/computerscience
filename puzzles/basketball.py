"""
There are 8 basketballs and 1 scale. 7 of them weigh the same. 1 of them is heavier. 
How do you find the odd ball with only 2 weighs?
"""

"""
We place 3 balls on the left scale with 3 on the right. Of the three scenarios (1) 
If the left side is heavier, we know one of them on the left is the heavier ball. 
Then, we weigh the first and second ball on the left to determine which one it is 
or whether it's the one not weighed. (2) If the right side is heavier, then we perform
the same thing, but, instead, using the balls on the right. (3) If the two sides
are equal, we can deduce the heavier ball by weighing the remaining two balls. 
"""

a = [1, 2, 3, 4, 5, 6, 7, 8]

def solve():
    """
    Solve as dictated above.
    """
    if weight(a[:3]) > weight(a[3:6]): # scenario (1)
        if weight(a[0]) > weight(a[1]):
            return 0 
        elif weight(a[1]) > weight(a[0]):
            return 1
        else:
            return 2
    elif weight(a[:3]) < weight(a[3:6]): # scenario (2)
        if weight(a[3]) > weight(a[4]):
            return 3
        elif weight(a[3]) < weight(a[4]):
            return 4
        else:
            return 5
    elif weight(a[6]) > weight(a[7]):
        return 6
    else:
        return 7
       
