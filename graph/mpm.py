"""
Malhotra, Pramodh Kumar, and Maheshwari (MPM) algorithm.
The algorithm has time complexity of (n^3). The algorithm operates in stages.
We construct the network N(f) [residual network] and from it we find the
auxiliary[layered] network AN(f). Then we find a maximal flow g in AN(f) and 
update flows in the main network
"""
