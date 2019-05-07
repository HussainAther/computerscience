"""
A very large bioinformatics department at a prominent university 
has a mix of 100 professors: some are honest and hard-working, 
while others are deceitful and do not like students. The honest 
professors always tell the truth, but the deceitful ones sometimes 
tell the truth and sometimes lie. You can ask any professors the 
following question about any other professor: "Professor Y, is 
Professor X honest?" Professor Y will answer with either "yes" or "no." 
Design an algorithm that, with no more than 198 questions, would allow 
you to figure out which of the 100 professors are honest (thus identifying 
possible research advisors). It is known that there are more honest than dishonest professors.
"""

honest = []
dishonest = []

def solve(a):
    """
    For a number of professors a, deduce which professors are honest.
    """
    pair = []
    while len(honest) + len(dishonst) != a:
        for i in a:
            if pair == []:
                pair.append(a)
            elif len(pair) == 1:
                pair.append(a) # for each pair 
                if question(pair) == (h, h): # ask both of them the opinion of each other. if the answer is (honest, honest)
                    honest.append(pair[0])
                    honest.append(pair[1])
    return honest, dishonest
               
       
