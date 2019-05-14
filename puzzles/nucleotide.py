"""
Two players play the following game with a nucleotide sequence of length n. 
At every turn a player may delete either one or two nucleotides from the sequence. 
The player who deletes the last letter wins. Who will win? Describe the winning 
strategy for each n.
"""

def nucleotide(n):
    """
    For a starting length n, player one will definitely win. Player one 
    must 
