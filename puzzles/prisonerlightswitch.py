import random

"""
One hundred prisoners are given the following challenge which if they solve will set them free. 
They are allowed to convene before the challenge and come up with a shared strategy. 
Once they decide on a strategy, the challenge starts and they will all be placed in solitary confinement with 
no way of communicating with each other, except that at every hour of the clock one of the 
prisoners will be picked at random and placed in a room with a single light and its switch, 
which the prisoner can either turn on or off. The light will be turned off at the beginning, 
before any prisoners enter the room. The goal is for one of the prisoners to know for certain 
that all the prisoners have visited the room with the light switch. As soon as one prisoner 
can deduce that with certainty, he can announce it and the prisoners win the challenge.
"""

switch = False # light switch Boolean

class Prisoner(object):
    visited = False
   
    def visit(self):
        self.visited True
        return False
    
class 
