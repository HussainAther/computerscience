"""
General algorithm for hierchical clustering analysis.
"""

# Returns minimum difference between any pair
def findMinDiff(arr, m):
  
    # Sort array in non-decreasing order
    arr = sorted(arr)
  
    # Initialize difference as infinite
    diff = 10**20
  
    # Find the min diff by comparing adjacent
    # pairs in sorted array
    min_pair = []
    for i in range(m-1):
        if arr[i+1] - arr[i] < diff:
            diff = arr[i+1] - arr[i]
            min_pair = (arr[i+1], arr[i])
  
    # Return min diff
    return diff, min_pair

# Computer the distance between a cluster and a list of clusters
def computeDistance(c, arr):
    distances = []
    for i in arr:
        distances.append(i-c)
    return distances

# Construct a graph T by assigning an isolated vertex to each cluster
def hca(d, n): # n is the number of clusters to form. # d is the data itself in dict form. the keys are the clusters and the items are the elements of them.
    T = {} # graph to consctruct and output. keys are clusters and items are elements.
    while n > 1:
        diff, closest = findMinDiff(d.keys(), len(d)) # Find the two closest clusters by computing distances and returning the minimum distance and corresponding pair.
        cluster = (closest, d[closest[0]], closest[1]]) # Create a new cluster merged from those two closest clusters with the elements of them.
        cluster_distances = computeDistance(cluster[0], d.keys()) # Calculate the distances between the newly merged cluster and the other clusters.
        T[cluster] = d[closest[0], closest[1]] # Add a new vertex cluster and connect it to the vertices that we have merged.
        del d[closest[0]] # Delete the merged clusters from the dictionary d
        del d[closest[1]]
        d[cluster] = cluster_distances # Create a new cluster in the dictionary with the calcualted distances.
        n -= 1 # Work on the next cluster
    return T # Return the newly constructed graph of clusters.
