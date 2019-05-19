"""
You are given n activities with their start and finish times. Select the maximum number 
of activities that can be performed by a single person, assuming that a person can only 
work on a single activity at a time.

Assume activities are sorted according to finish time. Print a maximum set
of activities that can be done by a single person with n total nunmber of 
activities, s array of start times, and f array of finish times.
"""

def maxact(s, f ):
    """
    For start s and finish f times, find the max number of activities. 
    """ 
    n = len(f) 
    i = 0
    for j in xrange(n): 
        if s[j] >= f[i]: 
            print(j) 
            i = j 
