"""
In 1879, Lewis Carroll proposed the following puzzle to the readers of Vanity Fair: 
transform one English word into another by going through a series of intermediate English 
words, where each word in the sequence differs from the next by only one substitution. 
To transform head into tail one can use four intermediates: head → heal → teal → tell → tall → tail. 
We say that two words v and w are equivalent if v can be transformed into w by 
substituting individual letters in such a way that all intermediate words are English words present in an English dictionary.
"""

def eqwords(a, b):
    """
    Given two words a and b, find out whether they're equivalent.
    We may find the Lewis Carroll distance between a and b as the 
    smallest number of substitutions needed to transform a into v
    with all intermediate words still being words in the dictionary.
    If a and b are not equivalent, return False.
    """
