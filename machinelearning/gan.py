import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

"""
Generative adversarial neural network (GAN).
"""

def get_distribution_sampler(mu, sigma):
    """
    Generate input distribution Gaussian.
    """
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))

def get_generator_input_sampler():
    """
    Input sampler for choosing from the distribution.
    """
    return lambda m, n: torch.rand(m, n)

def extract(v):
    """
    Extract values to a list.
    """
    return v.data.storage().tolist()

def stats(d):
    """
    Mean and standard deviation.
    """
    return [np.mean(d), np.std(d)]

def get_moments(d):
    """
    Return the first 4 moments of the data provided.
    """
    mean = torch.mean(d)
    diffs = d - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
    final = torch.cat((mean.reshape(1,), std.reshape(1,), skews.reshape(1,), kurtoses.reshape(1,)))
    return final

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(x))

# send four moments to the generator.
(name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 4)

# Data params
data_mean = 4 # Mean
data_stddev = 1.25 # Standard deviation
g_input_size = 1 # Random noise dimension coming into generator, per output vector
g_hidden_size = 5 # Generator complexity
g_output_size = 1 # Size of generated output vector
d_input_size = 500 # Minibatch size - cardinality of distributions
d_hidden_size = 10 # Discriminator complexity
d_output_size = 1 # Single dimension for 'real' vs. 'fake' classification
minibatch_size = d_input_size

d_learning_rate = 1e-3
g_learning_rate = 1e-3
sgd_momentum = 0.9

num_epochs = 5000
print_interval = 100
d_steps = 20
g_steps = 20

dfe, dre, ge = 0, 0, 0
d_real_data, d_fake_data, g_fake_data = None, None, None

discriminator_activation_function = torch.sigmoid
generator_activation_function = torch.tanh

d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()
G = Generator(input_size=g_input_size,
              hidden_size=g_hidden_size,
              output_size=g_output_size,
              f=generator_activation_function)
D = Discriminator(input_size=d_input_func(d_input_size),
                  hidden_size=d_hidden_size,
                  output_size=d_output_size,
                  f=discriminator_activation_function)
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)
g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)



print("Plotting the generated distribution...")
values = extract(g_fake_data)
print(" Values: %s" % (str(values)))
plt.hist(values, bins=50)
plt.xlabel('Value')
plt.ylabel('Count')
plt.title("Histogram of Generated Distribution")
plt.grid(True)
plt.show()
