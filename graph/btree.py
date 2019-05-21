import numbers

"""
B-tree (b tree btree) implementation.
"""

class BTreeSet(object):
    """
    The degree is the minimum number of children each non-root internal node must have.
    """
    def __init__(self, degree, coll=None):
        """
        Initialize the tree.
        """
	if not isinstance(degree, numbers.Integral):
		raise TypeError()
	if degree < 2:
		raise ValueError("Degree must be at least 2")
	self.minkeys = degree - 1      # At least 1, equal to degree-1
	self.maxkeys = degree * 2 - 1  # At least 3, odd number, equal to minkeys*2+1
	
	self.clear()
	if coll is not None:
		for obj in coll:
			self.add(obj)

    def __len__(self):
	return self.size
	
	
    def clear(self):
	self.root = BTreeSet.Node(self.maxkeys, True)
	self.size = 0
	
    def __contains__(self, obj):
	# Walk down the tree
	node = self.root
	while True:
	    found, index = node.search(obj)
	    if found:
		return True
    	    elif node.is_leaf():
		return False
	    else:  # Internal node
		node = node.children[index]

    def add(self, obj):
        """
        Special preprocessing to split root node
        """
	root = self.root
	if len(root.keys) == self.maxkeys:
	    child = root
	    self.root = root = BTreeSet.Node(self.maxkeys, False)  # Increment tree height
	    root.children.append(child)
	    root.split_child(self.minkeys, self.maxkeys, 0)
	# Walk down the tree
	node = root
	while True:
	    # Search for index in current node
	    assert len(node.keys) < self.maxkeys
	    assert node is root or len(node.keys) >= self.minkeys
	    found, index = node.search(obj)
	    if found:
	        return  # Key already exists in tree
	    if node.is_leaf():  # Simple insertion into leaf
                node.keys.insert(index, obj)
		self.size += 1
		return  # Successfully added
	    else:  # Handle internal node
		child = node.children[index]
		if len(child.keys) == self.maxkeys:  # Split child node
	            node.split_child(self.minkeys, self.maxkeys, index)
		    if obj == node.keys[index]:
	                return  # Key already exists in tree
		    elif obj > node.keys[index]:
		        child = node.children[index + 1]
		node = child
    def remove(self, obj):
	if not self._remove(obj):
	    raise KeyError(str(obj))
    def discard(self, obj):
	self._remove(obj)
	
