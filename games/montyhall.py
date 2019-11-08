from thinkbayes import Pmf

"""
Bayesian framework for Monty Hall problem.
Use probability mass function (Pmf).
"""

class Monty(Pmf):
    """
    Test various hypotheses in the problem. 
    """
    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()
        hypos = "ABC"
        pmf = Monty(hypos)
        data = "B"
        pmf.Update(data)
    
    def Update(self, data):
        """
        Update hte hypotheses given data.
        """ 
