import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV

"""
Build an image classifier
"""

# Load data.
image_train = pd.read_csv("image_train_data.csv")
image_test = pd.read_csv("image_test_data.csv")

# Take a look at the data.
image_train["image"].head()

# Train.
raw_pixel_model = LogisticRegressionCV()
le = preprocessing.LabelEncoder()
image_train["image_array"] = image_train["image_array"].apply(lambda x :[int(i) for i in x[1:-1].split(" ")])
image_test["image_array"] = image_test["image_array"].apply(lambda x :[int(i) for i in x[1:-1].split(" ")])
