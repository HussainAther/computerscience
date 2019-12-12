"""
The   fusc   integer sequence is defined as:
+ fusc(0) = 0
+ fusc(1) = 1
+ for n>1, the nth term is defined as:
+ + if n is even;
+ + + fusc(n) = fusc(n/2)
+ + if n is odd;     
+ + + fusc(n) = fusc((n-1)/2) + fusc((n+1)/2)
"""

def fusc(i):
    """
    Fusc sequence
    """
    def go(n):
        """
        Move to the next in the sequence. 
        """
        if 0 == n:
            return (1, 0)
        else:
            x, y = go(n // 2)
            if n % 2 == 0:
                return (x+y, y)
            else:
                return (x, x+y)
    if 1 > i:
        return 0
    else:
        return go(i-1)[0]
