import numpy as np
import enchant
d = enchant.Dict("en_US")

"""
In 1879, Lewis Carroll proposed the following puzzle to the readers of Vanity Fair: 
transform one English word into another by going through a series of intermediate English 
words, where each word in the sequence differs from the next by only one substitution. 
To transform head into tail one can use four intermediates: head → heal → teal → tell → tall → tail. 
We say that two words v and w are equivalent if v can be transformed into w by 
substituting individual letters in such a way that all intermediate words are English words 
present in an English dictionary.

Use enchant to check if words are in the dictionary like this d.check("Hello").
"""

alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"] 

def eqwords(a, b, s=0):
    """
    Given two words a and b, find out whether they're equivalent.
    We may find the Lewis Carroll distance between a and b as the 
    smallest number of substitutions needed to transform a into v
    with all intermediate words still being words in the dictionary.
    If a and b are not equivalent, return False. s is the number of 
    substitutions.
    """
    for i, j in enumerate(a): # for each index and letter in a
        p = [] # possible substitutions
        for k in alphabet: # for each alphabet letter we may substitute
            if j != k: # if we can substitute this letter for the letter in a
                tempa = a[:i] + k + s[i + 1:]  
                if d.check(tempa) and tempa != b:
                    p.append(tempa)
                    s += 1
                elif d.check(tempa) and tempa == b:
                    return s
            if p != []:
                for t in p:
                    eqwords(t, b) 
    return False
