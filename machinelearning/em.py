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

# Load dataset.
wiki = pd.read_csv("people_wiki.csv")

def load_sparse_csr(filename):
    """
    Load the dataset as instructed.
    """
    loader = np.load(filename)
    data = loader["data"]
    indices = loader["indices"]
    indptr = loader["indptr"]
    shape = loader["shape"]
    
    return csr_matrix( (data, indices, indptr), shape)

tf_idf = load_sparse_csr("4_tf_idf.npz")  # NOT people_wiki_tf_idf.npz
map_index_to_word = pd.read_json("4_map_index_to_word.json",typ="series")  # NOT people_wiki_map_index_to_word.gl
