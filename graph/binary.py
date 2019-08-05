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

    def subtreefirstposition(self, p):
        """
        Return position of first item in subtree rooted at p.
        """
        walk = p 
        while self.left(walk) is not None:
            walk = self.left(walk)
        return walk

    def subtreelastposition(self, p):
        """
        Return position of last item in subtree rooted at p.
        """
        walk = p
        while self.right(walk) is not None:
            walk = self.right(walk)
        return walk
    
    def first(self):
        """
        Return the first position in the tree.
        """
        return self.subtreefirstposition(self.root()) if len(self) > 0 else None

    def last(self):
        """
        Return the last position in the tree.
        """
        return self.subtreelastposition(self.root()) if len(self) > 0 else None

    def before(self, p):
        """
        Return the position just before p in the natural order.
        """
        self.validate(p)
        if self.left(p):
            return self.subterelastposition(self.left(p))
        else:
            walk = p
            above = self.parent(walk)
            while above is not None and walk == self.left(above):
                walk = above
                above = self.parent(walk)
            return above

    def after(self, p):
        """
        Return the position just after p in the natural order.
        """
        self.validate(p)
        if self.right(p):
            return self.subterelastposition(self.right(p))
        else:
            walk = p
            above = self.parent(walk)
            while above is not None and walk == self.right(above):
                walk = above
                above = self.parent(walk)
            return above

    def findposition(self, k):
        """
        Return position with key k.
        """
        if self.isempty():
            return None
        else:
            p = self.subtreesearch(self.root(), k)
            self.rebalanceaccess(p)
            return p

    def findmin(self):
        """
        Return (key, value) pair with minimum key.
        """
        if self.isempty():
            return None
        else:
            p = self.first()
            return (p.key(), p.value())

    def findge(self, k):
        """
        Return (key, value) pair with least key greater than or equal to k.
        """
        if self.isempty():
            return None
        else:
            p = self.findposition(k)
            if p.key() < k:
                p = self.after(p)
            return (p.key(), p.value()) if p is not None else None
