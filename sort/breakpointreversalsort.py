import random

"""
Use the breakpoints to sort for reversals.
"""

def makePermutation(n):
    """
    Generates a random permuation of the numbers 1..n-1 
    sandwiched between 0 and n. 
    """
    seq = range(1,n)
    random.shuffle(seq)
    return [0] + seq + [n]

def hasBreakpoints(seq):
    """
    Returns True if sequnces in not strictly increasing by 1. 
    """
    for i in xrange(1, len(seq)):
        if (seq[i] != seq[i-1] + 1):
            return True
    return False

def getStrips(seq):
    """
    Find contained intervals where sequence is ordered, and return intervals
    in as lists, increasing and decreasing. Single elements are considered
    decreasing. "Contained" excludes the first and last interval. 
    """
    deltas = [seq[i+1] - seq[i] for i in xrange(len(seq)-1)]
    increasing = list()
    decreasing = list()
    start = 0
    for i, diff in enumerate(deltas):
        if (abs(diff) == 1) and (diff == deltas[start]):
            continue
        if (start > 0):
            if deltas[start] == 1:
                increasing.append((start, i+1))
            else:
                decreasing.append((start, i+1))
        start = i+1
    return increasing, decreasing

def pickReversal(seq, strips):
    """
    Test each decreasing interval to see if it leads to a reversal that
    removes two breakpoints, otherwise, return a reversal that removes only one. 
    """
    reversal = (-1, None)
    left = [i for i, j in strips]
    right = [j for i, j in strips]
    for i in left:
        for j in right:
            if (i >= j-1):
                # skip invalid intervals and
                # those with only one element
                continue
            breakpointsRemoved = 0
            if (abs(seq[j-1] - seq[i-1]) == 1):
                # reversal will remove left breakpoint
                breakpointsRemoved += 1
            if (abs(seq[j] - seq[i]) == 1):
                # reversal will remove right breakpoint
                breakpointsRemoved += 1
            if (breakpointsRemoved > reversal[0]):
                reversal = (breakpointsRemoved, (i,j))
    print "%d:" % reversal[0],
    return reversal[1]

def doReversal(seq,(i, j)):
    """
    Perform the reversal.
    """
    return seq[:i] + [element for element in reversed(seq[i:j])] + seq[j:]

def improvedBreakpointReversalSort(seq):
    while hasBreakpoints(seq):
        increasing, decreasing = getStrips(seq)
        if len(decreasing) > 0:
            reversal = pickReversal(seq, increasing+decreasing)
        else:
            print "0:",
            reversal = increasing[0]
        print seq, "reversal", reversal
        seq = doReversal(seq,reversal)
    print seq, "Sorted"
    return

if __name__ == "__main__":
    while True:
        input = raw_input("Enter a list, the size of list, or 0 to exit:")
        if (input.find(",") > 0):
            L = [int(v) for v in input.split(",")]
        else:
            n = int(input)
            if (n == 0):
                break
            L = makePermutation(n)
        improvedBreakpointReversalSort(L)
