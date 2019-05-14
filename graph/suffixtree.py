"""
Build a suffix tree. 
"""

class SuffixNode:
     """
     Node class for constructing node-based graphs.
     """
     def __init__(self, suffix_link = None):
         """
         Initialize the node with its corresponding suffix links.
         """
         self.children = {}
         if suffix_link is not None:
             self.suffix_link = suffix_link
         else:
             self.suffix_link = self

     def add_link(self, c, v):
          """
          Link this node to node v via string c.
          """
          self.children[c] = v
