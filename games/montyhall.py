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
        Update the hypotheses given data.
        """
        for hypo in self.Values():
            like = self.likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()

    def Likelihood(self, data, hypo):
        """
        Return the likelihood of a person.
        """
        if hypo == data:
            return 0
        elif hypo == "A":
            return .5
        else:
            return 1
        for hypo, prob in pmf.Items():
            print(hypo, prob)
