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
