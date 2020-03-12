import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

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
train_image_array = [i for i in image_train["image_array"].values ]
train_y = le.fit_transform(image_train.label)

# Fit.
raw_pixel_model.fit(train_image_array,train_y)

# Predict.
test_image_array =  [i for i in image_test["image_array"].values]
test_y = le.transform(image_test.label)
le.inverse_transform(raw_pixel_model.predict(test_image_array[0:3]))

# Evaluate.
true_label = le.transform(image_test["label"])
raw_pred_label = raw_pixel_model.predict(test_image_array)
accuracy_score(true_label,raw_pred_label)

# Very poor accuracy (~.438)

# Use deep learning (deeplearning) to improve the accuracy
len(image_train)

image_train["deep_features"]=image_train["deep_features"].apply(lambda x:[float(i) for i in x[1:-1].split(" ")])
image_test["deep_features"]=image_test["deep_features"].apply(lambda x:[float(i) for i in x[1:-1].split(" ")])
train_deep_features = [i for i in image_train["deep_features"].values]
deep_features_model = LogisticRegressionCV()
deep_features_model.fit(train_deep_features,train_y)

# Apply deep features model to first few images of test set.
test_deep_features = [i for i in image_test["deep_features"].values]
le.inverse_transform(deep_features_model.predict(test_deep_features[0:3]))

# Compute accuracy.
deep_pred = deep_features_model.predict(test_deep_features)
true_label =le.transform(image_test['label'])
accuracy_score(true_label,deep_pred)

# ~.808
