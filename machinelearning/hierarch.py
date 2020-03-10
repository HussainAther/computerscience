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

# Load the dataset.
wiki = pd.read_csv("people_wiki.csv")

def load_sparse_csr(filename):
    """
    Extract TF-IDF features (term frequency–inverse document frequency).
    """ 
    loader = np.load(filename)
    data = loader["data"]
    indices = loader["indices"]
    indptr = loader["indptr"]
    shape = loader["shape"]
    return csr_matrix( (data, indices, indptr), shape)

tf_idf = load_sparse_csr("people_wiki_tf_idf.npz")
map_index_to_word = pd.read_json("people_wiki_map_index_to_word.json", typ="series")
