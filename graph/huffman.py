import heapq
import os

"""
Huffman coding is a lossless data compression technique. It uses variable-length
codes to the input characters based on the frequencies of their occurence. The
most frequent character has the smallest length code.
"""

class HeapNode:
    """
    Heap node class for constructing the graph.
    """
