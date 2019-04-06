
"""
Having solved the 8-queens problem, we turn our attention to solving the N-queens
problem for arbitrary N. That is, we need to place N queens on an N × N board such that
no pair of queens attack each other.

Let’s now assume that we are not allowed to write nested for loops (or other types of
loops) that have a degree of nesting more than 2. You might say that this is an artificial
constraint, but not only is the deeply nested 8-queens code aesthetically displeasing, but
the code is also not general. If you wanted to write a program to solve the N-queens
problem for N up to say 20, you would have to write functions to solve 4-queens (with 4
nested loops), 5-queens (with 5 nested loops), all the way to 20-queens (with 20 nested
loops!), and invoke the appropriate function depending on the actual value of N when the
code is run. What happens if you then want a solution to the 21-queens problem?

We will need to use recursion to solve the general N-queens problem. Recursion occurs
when something is defined in terms of itself. The most common application of recursion
in programming is where a function being defined is applied within its own definition.

Let's get recursive in here.
"""

def nQueens(size):
    """
    Initialize empty board and call the recursive N-queens
    procedure to print the returned solution.
    """
    board = [-1] * size
    rQueens(board, 0, size)
    prettyPrint(board)

def prettyPrint(board):
    """
    Print the board.
    """
    size = len(board)
    for i in range(size):
        for j in range(size):
            if board[j] == i:
                queen = j
        row = ". "* queen + "Q " + ". " * (size - queen - 1)
        print (row)

    return

def noConflicts(board, current):
    """
    Check for no conflicts.
    """
    for i in range(current):
        if (board[i] == board[current]):
            return False
        if (current - i == abs(board[current] - board[i])):
            return False
    return True


def rQueens(board, current, size):
    """
    Place queen on a given column so it doesn't
    conflictw ith the existing queens and then call
    itself recursively to place subseqeunt queens till the requisitie number of queens
    are placed.
    """
    if (current == size):
        return True
    else:
        for i in range(size):
            board[current] = i
            if (noConflicts(board, current)):
                done = rQueens(board, current + 1, size)
                if (done):
                    return True
        return False
