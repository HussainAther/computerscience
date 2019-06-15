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

# Data params
data_mean = 4
data_stddev = 1.25

print("Plotting the generated distribution...")
values = extract(g_fake_data)
print(" Values: %s" % (str(values)))
plt.hist(values, bins=50)
plt.xlabel('Value')
plt.ylabel('Count')
plt.title("Histogram of Generated Distribution")
plt.grid(True)
plt.show()
