"""
Binary tree.
"""

class TreeMap(LinkedBinaryTree, MapBase):
    """
    Sorted map implementation using a binary search tree. 
    """
    class Position(LinkedBinaryTree.Position):
        def key(self):
            """
            Return key of map's key-value pair.
            """
            return self.element().key
        def value(self):
            """
            Return value of map's key-value pair.
            """
            return self.element().value

def subtreesearch(self, p, k):
    """
    Return Posiiton of p's subtree having key k or last node searched.
    """
    if k == p.key():
        return p
    elif k < p.key():
        if self.left(p) is not None:
            return self.subtreesearch(self.left(p), k)
    else:
        if self.right(p) is not None:
            return self.subtreesearch(self.right(p), k)
    return p
