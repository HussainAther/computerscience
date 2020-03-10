import pandas as pd                                     
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans                
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

"""
Hierarchical clustering from University of Washington Coursera's 
"Machine Learning: Clustering & Retrieval"
"""
