from itertools import takewhile, tee, islice
from collections import deque
from fractions import gcd

"""
In number theory, the Stern–Brocot tree is an infinite complete binary tree 
in which the vertices correspond one-for-one to the positive rational numbers, 
whose values are ordered from the left to the right as in a search tree.

stern brocot Stern Brocot
"""

def stern_brocot():
    """
    We iterate to produce successive members of the sequence. We check the gcd's
    (Greatest common denominators greatest denominators) to find two iterators
    that we tee off from the one stream with the second advanced by one value with 
    the next() call.
    """
    sb = deque([1, 1])
    while True:
        sb += [sb[0] + sb[1], sb[1]]
        yield sb.popleft()

# Generate s-b numbers
[s for _, s in zip(range(15), stern_brocot())]
[1 + sum(1 for i in takewhile(lambda x: x != occur, stern_brocot()))
     for occur in (list(range(1, 11)) + [100])]

# Test verify our tree
prev, this = tee(stern_brocot(), 2)
next(this) # should return 1
all(gcd(p, t) == 1 for p, t in islice(zip(prev, this), 1000)) # should return True
