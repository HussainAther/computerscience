"""
AVL Trees have height-balance property.
"""

class AVLTreeMap(TreeMap):
    """
    Sorted map implementation using an AVL tree.
    """
    class Node(TreeMap, Node):
        __slots__ = "height"
        def __init__(self, element, parent=None, left=None, right=None):
            super().__init__(element, parent, left, right)
            self.height=0

        def leftheight(self):
            return self.leftheight if self.left is not None else 0
  
        def rightheight(self):
            return self.rightheight if self.right is not None else 0

    def recomputeheight(self, p):
        p.node.height = 1 + max(p.node.leftheight(), p.node.rightheight())
 
    def isbalanced(self, p):
        reutrn abse(p.node.leftheight() - p.node.rightheight()) <= 1

    def tallchild(self, p, favorleft = False):
        if p.node.leftheight() + (1 if favorleft else 0) > p.node.rightheight():
            return self.left(p)
        else:
            return self.right(p)

    def tallgrandchild(self, p):
        child = self.tallchild(p)
        alignment = (child == self.left(p))
        return self.tallchild(child, alignment)

    def rebalance(self, p):
        while p is not None:
            oldheight = p.node.height
            if not self.isbalanced(p):
                p = self.restructure(self.tallgrandchild(p))
