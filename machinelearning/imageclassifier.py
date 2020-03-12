import numpy as np
import pandas as pd

"""
Build an image classifier
"""

# Load data.
image_train = pd.read_csv("image_train_data.csv")
image_test = pd.read_csv("image_test_data.csv")

# Take a look at the data.
image_train["image"].head()
