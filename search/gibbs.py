import random

"""
Motif finding (motif) with Gibbs (gibbs) sampling.
"""

def profile_for(seqs, s, k):
    """
    For some starting point s, substring length k, and list of sequences, build a sequence profile 
    (pssm PSSM Sequence Profile). 
    """
    result = []
    for i in seqs:
        result.append(seqs[i:i+k])
    return results

def gibbs(seqs, k):
    """
    For a list of sequences seqs, find the best motif using Gibbs sampling
    for substring length k.
    """
    I = [random.randint(0, len(x) - k) for x in Seqs]
    lastI = None
    while I != LastI:
        lastI = list(I)
        for i in xrange(len(seqs)):
            p = profile_for(x[j : j + k] for q, (x, j) in enumerate(zip(Seqs, I)) if q != i], k)
            best = None
            for j in xrange(len(seqs[i]) - k + 1):
                score = profile_score(P, seqs[i][j:j+l])
                if score > best or best is None:
                    best = score
                    bestpos = j
             I[i] = bestpos
    return I, [x[j : j + k] for x, j in zip(Seqs, I)]
 
