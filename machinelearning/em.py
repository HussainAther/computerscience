import numpy as np
import pandas as pd

from copy import deepcopy
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

"""
Fitting a diagonal covariance Gaussian mixture model to text data using 
expectation-maximization (EM) from University of Washington Coursera's
"Machine Learning: Clustering & Retrieval"
"""
