"""
Let’s say we have a whole bunch of people in line waiting to see a baseball game. They
are all hometown fans and each person has a team cap and is wearing it. However, not all
of them are wearing the caps in the same way – some are wearing their caps in the normal
way, and some are wearing them backwards.

People have different definitions of normal and backwards but you, the gatekeeper thinks
that the cap on the left below is being normally worn and the one on the right is being
worn backwards.

Input is a vector of F's and B's, in terms of forwards and backwards caps
Output is a set of commands (printed out) to get either all F's or all B's
Our goal is to perform this in the fewest number of commands.
"""

caps = ["F", "F", "B", "B", "B", "F", "B", "B", "B", "F", "F", "B", "F" ]
cap2 = ["F", "F", "B", "B", "B", "F", "B", "B", "B", "F", "F", "F", "F" ]

def pleaseConform(caps):
    """
    Why are we being so conformist? :thinking:
    """
    start = 0
    forward = 0
    backward = 0
    intervals = []

    for i in range(len(caps)): # find intervals in which caps are on in the same direction
        if caps[start] != caps[i]:
            # each interval is a tuple with 3 elements (start, end, type)
            intervals.append((start, i - 1, caps[start]))
            
            if caps[start] == "F":
                forward += 1
            else:
                backward += 1
            start = i

    intervals.append((start, len(caps) - 1, caps[start]))
    # add to last interval
    if caps[start] == "F":
        forward += 1
    else:
        backward += 1

     # switch it up
    if forward < backward:
        flip = "F"
    else:
        flip = "B"
    for t in intervals:
        if t[2] == flip:
            #Exercise: if t[0] == t[1] change the printing!
            if t[0] == t[1]:
                print("Person at position", t[0], "flip your cap!")
            else:
                print("People in positions", t[0], "through", t[1], "flip your caps!")

