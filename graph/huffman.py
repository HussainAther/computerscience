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
        freq = {}
        for c in text: # for each character
            if not c in freq:
                freq[c] = 0
            freq[c] += 1
        return freq
    
    def heap(self, freq):
        """
        For frequency dict freq, make heap.
        """
        for key in freq:
            node = HeapNode(key, freq[key])
            heapq.heappush(self.heap, node)
    
    def mergenodes(self):
        """
        Merge nodes using the heap.
        """
        while(len(self.heap)>1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)

    def codehelper(self, root, current_code):
        """
        Helper functions for creating the code.
        """
        if root == None:
            return
        elif root.char != None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return
        self.codehelper(root.left, current_code + "0")
        self.codehelper(root.right, current_code + "1")
