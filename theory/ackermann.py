"""
The Ackermann function uses a total computable function that is not primitive function.
We illustrate the Ackermann (ackermann) function also konwn as the Ackermann-Péter function 
(peter Peter) which uses two nonnegative integers m and n. We can uuse it optimize recursion.
"""

def ack(m, n):
    """
    Iterate through Ackermann function.
    """
    if m == 0:
        return n + 1
    elif n == 0:
        return A(m - 1, 1, s)
    n2 = ack(m, n - 1)
    return A(m - 1, n2, s)
