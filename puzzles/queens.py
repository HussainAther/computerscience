from itertools import permutations

"""
A lovely chess problem.

The 8-Queens problem on a chessboard corresponds to finding a placement of 8 queens
such that no queen attacks any other queen. This means that
1. No two queens can be on the same column.
2. No two queens can be on the same row.
3. No two queens can be on the same diagonal.

Systematic search.

This is how I play chess: https://www.youtube.com/watch?v=8ghGvbdlTDQ
"""

def noConflicts(board, current):
    """
    Check to make sure we're playing by the rules.
    """
    for i in range(current):
        if (board[i] == board[current]):
            return False
        if (current - i == abs(board[current] - board[i])):
            return False
    return True

def EightQueens(numsol, n=8):
    """
    Place 8 queens on a board so they don't break rules.
    """
    board = [-1] * n
    sol = 0
    # check by each unit
    for i in range(n):
        board[0] = i
        for j in range(n):
            board[1] = j
            if not noConflicts(board, 1):
                continue
            for k in range(n):
                board[2] = k
                if not noConflicts(board, 2):
                    continue
                    for l in range(n):
                        board[3] = l
                        if not noConflicts(board, 3):
                            continue
                            for m in range(n):
                                board[4] = m
                                if not noConflicts(board, 4):
                                    continue
                                for o in range(n):
                                    board[5] = o
                                    if not noConflicts(board, 5):
                                        continue
                                    for p in range(n):
                                        board[6] = p
                                        if not noConflicts(board, 6):
                                            continue
                                        for q in range(n):
                                            board[7] = q
                                            if noConflicts(board, 7):
                                                if sol < numsol:
                                                    print(board)
                                                    sol += 1
                                                else:
                                                    return
    return
    
EightQueens(7)

"""
Altenrate solution.
"""

n = 8
cols = range(n)
for vec in permutations(cols):
    if (n == len(set(vec[i]+i for i in cols))
          == len(set(vec[i]-i for i in cols))):
        print(vec)

