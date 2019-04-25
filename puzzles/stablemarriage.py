"""
The stable marriage problem or the stable matching problem (smp SMP).
Given n men and n women, where each person has ranked all members of the 
opposite sex in order of preference, marry the men and women together such 
that there are no two people of opposite sex who would both rather have each 
other than their current partners. When there are no such pairs of people, 
the set of marriages is deemed stable.
"""

def smp(m, w):
    """
    List of preferences for men m and women w. m should be an array of arrays with the array
    index for each women. w should be an array of arrays with the array index for each man.
    """
    freem = np.range(len(m)) # list of free men by array index
    freew = np.range(len(w)) # list of free women by array index
    while m: # while there are still eligible men
        for man in m:
            if m[man] != []
                w0 = m[man][0] # first woman on the first man's list that has not proposed 
                if w0 not in freew: # if w0 is free
       	            freew.remove(w0) # w0 engages to the man
                    freem.remove(m.index(man))
                else:
                    for woman in   
