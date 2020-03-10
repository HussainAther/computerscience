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

# Normalize the vectors.
tf_idf = normalize(tf_idf)

def bipartition(cluster, maxiter=400, num_runs=4, seed=None):
    """cluster: should be a dictionary containing the following keys
                * dataframe: original dataframe
                * matrix:    same data, in matrix format
                * centroid:  centroid for this particular cluster
    """
    data_matrix = cluster["matrix"]
    dataframe   = cluster["dataframe"]

    # Run k-means on the data matrix with k=2. We use scikit-learn here to simplify workflow.
    kmeans_model = KMeans(n_clusters=2, max_iter=maxiter, n_init=num_runs, random_state=seed, n_jobs=-1,verbose=1)
    kmeans_model.fit(data_matrix)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
    
    # Divide the data matrix into two parts using the cluster assignments.
    data_matrix_left_child, data_matrix_right_child = data_matrix[cluster_assignment==0], \
                                                      data_matrix[cluster_assignment==1]
    
    # Divide the dataframe into two parts, again using the cluster assignments.
    cluster_assignment_sa = np.array(cluster_assignment) # minor format conversion
    dataframe_left_child, dataframe_right_child     = dataframe[cluster_assignment_sa==0], \
                                                      dataframe[cluster_assignment_sa==1]
