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
    def __init__(self, char, freq):
        """
        Initialize the character frequencies in the graph.
        """
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __cmp__(self, other):
        """
        Compression function for outputting the compressed file.
        """
        if other == None:
            return -1
        elif not isinstance(other, HeapNode)):
            return -1
        return self.freq > other.freq

class HuffmanCoding:
    """
    Derive the Huffman codes.
    """
    def __init__(self, path):
        """
        Initialize the path and associated variables.
        """
        self.path = path
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
    
    def freqdict(self, text):
        """
        Initialize a frequency dictionary for text containing the characters.
        """
