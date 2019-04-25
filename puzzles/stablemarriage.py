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
    matches = {} # dictionary of matches with key as woman and value as man
    while m: # while there are still eligible men
        for man in m:
            if m[man] != []
                w0 = m[man][0] # first woman on the first man's list that has not proposed 
                if w0 not in freew: # if w0 is free
       	            freew.remove(w0) # w0 engages to the man. remove her from free women
                    w[w0].remove(man) # remove the man from the woman's list
                    freem.remove(m.index(man)) # remove the man from free mwn
                    m[man].remove(w0) # remove the woman the man's list
                    matches[w0] = man # add to dictionary of matches
                else: # if w0 is taken then it means that a pair (m0, w0) already exists
                    if w[w0].index(man) > w[w0].index(matches[w0]) # if the woman prefers this new man to her current match
                        freem.append(matches[w0]) # her current match becomes free
                        matches[w0] = man # she trades and gets the new man
    
