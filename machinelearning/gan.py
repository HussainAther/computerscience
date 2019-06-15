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
