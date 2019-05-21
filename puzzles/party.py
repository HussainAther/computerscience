"""
There is a party to celebrate celebrities that you get to attend because you won a ticket at
your office lottery. Because of the high demand for tickets you only get to stay for one
hour but you get to pick which one since you received a special ticket. You have access
to a schedule that lists when exactly each celebrity is going to attend the party. You want
to get as many pictures with celebrities as possible to improve your social standing. This
means you wish to go for the hour when you get to hob-nob with the maximum number of
celebrities and get selfies with each of them.

We are given a list of intervals that correspond to when each celebrity comes and goes.
Assume that these intervals are initially inclusive, where the first and second values correspond to hours.
That is, the interval is closed on the left hand side and open on the right hand side.
This just means that the celebrity will be partying on and through the th hour, but will
have left when the th hour begins. So even if you arrive on dot on the th hour, you will
miss this particular celebrity.

When is the best time to attend the party? That is, which hour should you go to?

Given a list of intervals when celebrities will be at the party
Output is the time that you want to go the party when the maximum number of
celebrities are still there.

Clever algorithm that will work with fractional times.
"""

sched = [(6, 8), (6, 12), (6, 7), (7, 8), (7, 10), (8, 9), (8, 10), (9, 12),
            (9, 10), (10, 11), (10, 12), (11, 12)]
sched2 = [(6.0, 8.0), (6.5, 12.0), (6.5, 7.0), (7.0, 8.0), (7.5, 10.0), (8.0, 9.0),
          (8.0, 10.0), (9.0, 12.0), (9.5, 10.0), (10.0, 11.0), (10.0, 12.0), (11.0, 12.0)]
sched3 = [(6, 7), (7,9), (10, 11), (10, 12), (8, 10), (9, 11), (6, 8),
          (9, 10), (11, 12), (11, 13), (11, 14)]

def bestTimeToPartySmart(schedule, ystart, yend):
    """
    Edit the schedule to find party times.
    """
    times = []
    for c in schedule:
        times.append((c[0], "start"))
        times.append((c[1], "end"))

    sortlist(times)
    maxcount, time = chooseTimeConstrained(times, ystart, yend)
    print("The best time to attend the party is at", time,\
           "o\"clock", ":", maxcount, "celebrities will be attending!")

def sortlist(tlist):
    """
    Tuple method to sort.
    """
    for index in range(len(tlist)-1):
        ismall = index
        for i in range(index, len(tlist)):
            if tlist[ismall][0] > tlist[i][0] or \
               (tlist[ismall][0] == tlist[i][0] and \
                tlist[ismall][1] > tlist[i][1]):
                ismall = i
        tlist[index], tlist[ismall] = tlist[ismall], tlist[index]
    
    return


def chooseTimeConstrained(times, ystart, yend):
    """
    Use the contraint to choose times.
    """
    rcount = 0
    maxcount = 0
    time = 0
    for t in times: # max number of celebrities
        if t[1] == "start":
            rcount = rcount + 1
        elif t[1] == "end":
            rcount = rcount - 1
        if rcount > maxcount and t[0] >= ystart and t[0] < yend: # are you available at this time?
            maxcount = rcount
            time = t[0]

    return maxcount, time
