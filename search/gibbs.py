import random

"""
Motif finding (motif) with Gibbs (gibbs) sampling.
"""

def gibbs(seqs):
    """
    For a list of sequences seqs, find the best motif using Gibbs sampling.
    """
    k = len(seqs)
    I = [random.randint(0, len(x) - k) for x in Seqs]
    lastI = None
    while I != LastI:
        lastI = list(I)
        for i in xrange(len(seqs)):
            p = profile_for(x[j : j + k] for q, (x, j) in enumerate(zip(Seqs, I)) if q != i]) 
