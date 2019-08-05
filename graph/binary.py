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

    def findrange(self, start, stop):
        """
        Iterate all (key, value) pairs such that start <= key < stop.
        """
        if not self.isempty():
            if start is None:
                p = self.first()
            else:
                p = self.findposition(start)
                if p.key() < start:
                    p = self.after(p)
            while p is not None and (stop is None or p.key() < stop):
                yield (p.key(), p.value())
                p = self.after(p)
    
    def __getitem__(self, k):
        """
        Return value associtaed with key k.
        """
        if self.isempty():
            raise KeyError("Key Error: " + repr(k))
        else:
            p = self.subtreesearch(self.root(), k)
            self.rebalanceaccess(p)
            if k != p.key():
                raise KeyError("Key Error: " + repr(k))
            return p.value()

    def __setitem__(self, k, v):
        """
        Assign value v to key k, overwriting existing value if present.
        """
        if self.isempty():
            leaf = self.addroot(self.item(k, v))
        else:
            p = self.subtreesearch(self.root(), k)
            if p.key() == k:
                p.element().value = v
                self.rebalanceaccess(p)
                return
            else:
                item = self.item(k, v)
                if p.key() < k:
                    leaf = self.addright(p, item)
                else:
                    leaf = self.addleft(p, item)
        self.rebalanceinsert(leaf)

    def __iter__(self):
        """
        Generate an iteration of all keys in the map in order.
        """
        p = self.first()
        while p is not None:
            yield p.key()
            p = self.after(p)

    def delete(self, p):
        """
        Remove the item at given position.
        """
        self.validate(p)
        if self.left(p) and self.right(p):
            replacement = self.subtreelastposition(self.left(p))
            self.replace(p, replacement.element())
            p = replacement
        parent = self.parent(p)
        self.delete(p)
        self.rebalancedelete(parent)

