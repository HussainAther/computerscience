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

    def codehelper(self, root, currentcode):
        """
        Helper functions for creating the code.
        """
        if root == None:
            return
        elif root.char != None:
            self.codes[root.char] = currentcode
            self.reverse_mapping[currentcode] = root.char
            return
        self.codehelper(root.left, currentcode + "0")
        self.codehelper(root.right, currentcode + "1")
  
    def makecode(self):
        """
        Make the Huffman code.
        """
        root = heapq.heappop(self.heap)
        currentcode = ""
        self.codehelper(root, currentcode)

    def encodetext(self, text):
        """
        Get the encoded text.
        """
        encodedtext = ""
        for c in text:
            encodedtext += self.codes[c]
        return encodedtext

    def padencodedtext(self, encodedtext):
        """
        Pad the encoded text.
        """ 
        extrapadding = 8 - len(encodedtext) % 8
        for i in range(extrapadding):
            encodedtext += "0"
        paddedinfo = "{0:08b}".format(extrapadding)
        encodedtext = paddedinfo + encodedtext
        return encodedtext
  
    def bytearray(self, paddedencodedtext):
        """
        Create a byte array for the padded encoded text.
        """
        if len(paddedencodedtext) % 8 != 0:
            print("Not padded properly.")
            exit(0)
        b = bytearray()
        for i in range(0, len(paddedencodedtext), 8):
            byte = paddedencodedtext[i: i+8]
            b.append(int(byte, 2))
        return b

    def compress(self):
        """
        Compress a file.
        """
        filename, fileextension = os.path.splitext(self.path)
        outputpath = filename + ".bin"
        with open(self.path, "r+") as file, open(outputpath, "wb") as output:
            text = file.read()
            text = text.rstrip()
            freq = self.freqdict(text)
            self.heap(freq)
            self.mergenodes()
            self.makecode()
            encodedtext = self.encodetext(text)
            paddedencodedtext = self.adencodedtext(text)
            b = self.bytearraY(paddedencodedtext)
            output.write(bytes(b))
        return outputpath

    def removepadding(self, paddedencodedtext):
        """
        Remove padding for decompressing.
        """
        paddedinfo = paddedencodedtext[:8]
        extrapadding = int(paddedinfo, 2)
        paddedencodedtext = paddedencodedtext[8:]
        encodedtext = paddedencodedtext[:-1*extrapadding]
        return encodedtext
 
    def decodetext(self, encodedtext):
        """
        Decode text for decompression.
        """
        currentcode = ""
        decodedtext = ""
        for bit in encodedtext:
            currentcode += bit
            if currentcode in self.reversemapping:
                c = self.reversemapping[currentcode]
                decodedtext += c
                currentcode = ""
        return decodedtext
 
    def decompress(self, inputpath):
        """
        Decompress from input path.
        """
        filename, fileextension = os.path.splitext(self.path)
        outputpath = filename + "_decompressed.txt"
        with open(inputpath, "rb") as file, open(outputpath, "w") as output:
            bitstring = ""
            byte = file.read(1)
            while byte != "":
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, "0")
                bitstring += bits
                byte = file.read(1)
            encodedtext = self.removepadding(bitstring)
            decompressedtext = self.decodetext(encodedtext)
            output.write(decompressedtext)
        return outputpath
