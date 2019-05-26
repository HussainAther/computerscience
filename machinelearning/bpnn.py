import math
import random
import string

"""
Backpropagation Neural Networks
"""

random.seed(0) # Seed for reproducibility

def rand(a, b):
    """
    Return a random number a <= rand < b
    """
    return (b-a)*random.random() + a

def makeMatrix(I, J, fill=0.0):
    """
    Create a matrix.
    """
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def sigmoid(x):
    """
    Sigmoid function using tanh to approximate.
    """
    return math.tanh(x)

def dsigmoid(y):
    """
    Sigmoid function derivative
    """
    return 1.0-y*y

class NN:
    def __init__(self, ni, nh, no):
        """
        Number of input, hidden, and output nodes
        """
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        """
        Activations for nodes
        """
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        """
        Create weights
        """
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        """
        Set weight values
        """
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(0, 2.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(0, 2.0)
        
        """
        Last change in weights for momentum
        """
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError("wrong number of inputs")

        """
        Input activations
        """
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        """
        Hidden activations
        """
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        """
        Output activations
        """
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError("wrong number of target values")

        """
        Calculate error terms for output
        """
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        """
        Calculate error terms for hidden
        """
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        """
        Udate output weights
        """
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        """
        Update input weights
        """
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        """
        Calculate error
        """
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], "->", self.update(p[0]))

    def weights(self):
        print("Input weights:")
        for i in range(self.ni):
            print(self.wi[i])
        print("Output weights:")
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
        return error

def demo():
    """
    Teach network XOR function
    """
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    """
    Create a network with two input, two hidden, and one output nodes
    """
    n = NN(2, 2, 1)
    """
    Train it with some patterns
    """
    n.train(pat)
    """
    Test it
    """
    n.test(pat)

if __name__ == "__main__":
    demo()
