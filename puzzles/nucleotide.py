"""
Two players play the following game with a nucleotide sequence of length n. 
At every turn a player may delete either one or two nucleotides from the sequence. 
The player who deletes the last letter wins. Who will win? Describe the winning 
strategy for each n.
"""

def nucleotide(n):
    """
    For a starting length n, player one will definitely win. Player one 
    must force player two to always be given an odd number n. That way,
    when player two is given a 3-length nucleotide sequence, player one
    wins no matter what player two does. For this function, we assume the
    input is the length of the sequence player one has. Player one loses
    if he/she starts out with a nucleotide of length 3.
    """
    if n == 3:
        return False
    elif n == 2 or n == 1:
        return True
    elif n % 2 == 0:
        return n - 1
    else:
        return n - 2
          
