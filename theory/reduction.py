
"""
F reduces to G means, informally, that if you can olve G, you can solve F. Turing reduction can informally
be stated as F polyreduces to G means if oyu can solve G in polynomial time, you can solve F in polynomial time.

A decision problem F has a polynomial time mappign reduction to problem G if there exists a polytime program C that converts instances
of F into instances of G such taht C maps positive instances of F to positive instances of G, and negative instances of F to negative
instances of G.

Partition takes a set of integer weights as input but with no threshold values. It has the solution "yes" if the weights can be partitioned
into two sets of equal total weight, and has the value of "no" otherwise. Packing takes a set of integer weights w as input and receives two
additional integers L and H which are teh low and high threshold values. Its solution is "yes" if there's a subset of the weights that sums to a value
between L and H or "no" if it's impossible.
"""

def packing(weights, L, H):
    """
    Returns a feasible packing, a subset of the input weights whose total weight W lies between the low and high thresholds: L <= W <= H. If no
    feasiable packing exists, the solution is "no." Otherwise a soultion is a subset S of the weights that represents a feasible packing.
    """
    weights = sorted(weights, reverse=True)
    bins = []

    for item in weights:
        # Try to fit item into a bin
        for bin in bins:
            if bin.sum + item <= H and bin.sum + item >= L:
                #print 'Adding', item, 'to', bin
                bin.append(item)
                break
        else:
            # item didn't fit into any bin, start a new bin
            #print 'Making new bin for', item
            bin = Bin()
            bin.append(item)
            bins.append(bin)

    return bins

def partition(weights):
    """
    Same as packing but the weight of a feasible packing must equal exactly half the total
    weight of the packages
    """
    exactWeight = sum(weights)/2

    weights = sorted(weights, reverse=True)
    bins = []

    for item in weights:
        # Try to fit item into a bin
        for bin in bins:
            if bin.sum + item == exactWeight:
                #print 'Adding', item, 'to', bin
                bin.append(item)
                break
        else:
            # item didn't fit into any bin, start a new bin
            #print 'Making new bin for', item
            bin = Bin()
            bin.append(item)
            bins.append(bin)

    return bins


def convertPartitionToPacking(inString):
    """
    Polyreduction
    """
    weights = [int(x) for x in inString.split()]
    totalWeight = sum(weights)
    if totalWeight % 2 == 1:
        # if the total weight is odd, no partition is possible,
        # so return any negative instance of Packing
        return "0;1;1"
    else:
        # use thresholds that are half the total weight
        targetWeight = str(int(totalWeight / 2))
        return instring+";"targetWeight+";"targetWeight
