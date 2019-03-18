
"""
A handyman has a whole collection of nuts and bolts of different sizes in a bag. Each nut
is unique and has a corresponding unique bolt, but the disorganized handyman has
dumped them all into one bag and they are all mixed up. How best to “sort” these nuts
and attach each to its corresponding bolt?

Given n nuts and n bolts, the handyman can pick an arbitrary nut and try it with each bolt
and find the one that fits the nut. Then, he can put away the nut-bolt pair, and he has a
problem of size n – 1. This means that he has done n “comparisons” to reduce the
problem size by 1. n – 1 comparisons will then shrink the problem size to n – 2, and so
on. The total number of comparisons required is n + (n – 1) + (n – 2) + … + 1 =
n(n+1)/2. You could argue the last comparison is not required since there will only be one
nut and one bolt left, but we will call it a confirmation comparison.

Can one do better in terms of number of comparisons required? More concretely, can
one split the nuts and bolts up into two sets, each of half the size, so we have two
problems of size n/2 to work on? This way, if the handyman has a helper, they can work
in parallel. Of course, we could apply this strategy recursively to each of the problems of
size n/2 if there are additional kind people willing to help.

Can you think of a recursive Divide and Conquer strategy to solve the Nuts and Bolts
problem so you require significantly fewer than n(n+1)/2 comparisons when n is large?
"""

def pivotPartitionClever(lst, start, end):
    """
    Select a pivot and partitions the list into 3 sublists.
    """
    pivot = lst[end]
    bottom = start - 1
    top = end
    moves = 0

    done = False

    while not done:

        while not done:
            #Move rightward from left searching for element > pivot
            bottom += 1
            if bottom == top:
                done = True
                break
            if lst[bottom] > pivot:
                lst[top] = lst[bottom]
                moves += 1
                #print (lst, 'bottom =', bottom, 'top = ', top)
                break
    
        while not done:
            #Move leftward from right searching for element < pivot
            top -= 1
            if top == bottom:
                done = True
                break
            if lst[top] < pivot:
                lst[bottom] = lst[top]
                moves += 1
                #print (lst, 'bottom =', bottom, 'top = ', top)
                break

    lst[top] = pivot
    return top, moves

def quicksort(lst, start, end):
    """
    Quick sort implementation.
    """
    moves = 0
    if start < end:
        split, moves = pivotPartitionClever(lst, start, end)
        moves += quicksort(lst, start, split - 1)
        moves += quicksort(lst, split + 1, end)
    return moves
