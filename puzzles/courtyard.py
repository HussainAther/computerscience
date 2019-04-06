"""
Consider the following tiling problem. We have a courtyard with 2n × 2n squares and we
need to tile the courtyard using L-shaped tiles or trominoes. Each trominoe consists of
three square tiles attached to form an L shape as shown below.

Can this be done without spilling over the boundaries, breaking a tromino or having
overlapping trominoes? The answer is no, simply because 2^n × 2^n = 2^2n is not divisible
by 3, only by 2. However, if there is one square that can be left untiled, then 22n – 1 is
divisible by 3. Can you show this? We, therefore, have hope of properly tiling a 2^n × 2^n
courtyard with one square that we can leave untiled because, for example, there is a statue
of your favorite President on it.

We’ll call this square that can be left untiled the missing square.

Is there an algorithm that tiles any 2^n × 2^n courtyard with one missing square in an
arbitrary location? As an example, below is a 2^3 × 2^3 where the missing square is
marked Δ. Does the location of the missing square matter?
"""

def tileFourMissingYard(n, missing):
    """
    Given n in a 2^n x 2^n checkyard with a missing square at position (r, c),
    find tililng of yard with trominoes (L-shaped dominoes).
    """
    size = 2**n
    missingquad = []
    """
    Get the missing tiles.
    """
    for (r,c) in missing:
        missingquad.append(2*(r >= size//2) +(c >=size//2))
    count = 0
    
    for i in range(4):
        if i in missingquad:
            count +=1
    """
    This means yes.
    """
    if count == 4:
        return True
    """
    Do we have the appropriate arrangement? Check.
    """
    for (r,c) in missing:
        if (((r+1,c) in missing and (r,c+1) in missing) or ((r-1,c) in missing and (r,c+1) in missing)
        or ((r+1,c) in missing and (r,c-1) in missing) or ((r-1,c) in missing and (r,c-1)in missing)):
            return True
    """
    This means no.
    """
    return False

tileFourMissingYard(3, [(4, 4), (1, 1), (2, 1), (1, 2)])
tileFourMissingYard(3, [(4, 4), (3, 1), (2, 1), (1, 2)])
tileFourMissingYard(3, [(3, 7), (4, 4), (4, 6), (4, 7)])
