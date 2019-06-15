import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

"""
Generative adversarial neural network (GAN).
"""

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
