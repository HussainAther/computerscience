class Expression:
    """
    This abstract class does nothing on its own.
    """
    pass

class Sum(list, Expression):
    """
    A Sum acts just like a list in almost all regards, except that this code
    can tell it is a Sum using isinstance(), and we add useful methods
    such as simplify(). You can index into a sum like a list, as in term = sum[0].
    You can iterate over a sum with "for term in sum:". You can convert a sum to an 
    ordinary list with the list() constructor: the_list = list(the_sum)
    You can convert an ordinary list to a sum with the Sum() constructor:
    the_sum = Sum(the_list)
    """
    def __repr__(self):
        return "Sum(%s)" % list.__repr__(self)
    
    def simplify(self):
        """
        This is the starting point for the task you need to perform. It
        removes unnecessary nesting and applies the associative law.
        """
        terms = self.flatten()
        if len(terms) == 1:
            return simplify_if_possible(terms[0])
        else:
            return Sum([simplify_if_possible(term) for term in terms]).flatten()

    def flatten(self):
        """Simplifies nested sums."""
        terms = []
        for term in self:
            if isinstance(term, Sum):
                terms += list(term)
            else:
                terms.append(term)
        return Sum(terms)


class Product(list, Expression):
    """
    See the documentation above for Sum. A Product acts almost exactly
    like a list, and can be converted to and from a list when necessary.
    """
    def __repr__(self):
        return "Product(%s)" % list.__repr__(self)
    
    def simplify(self):
        """
        To simplify a product, we need to multiply all its factors together
        while taking things like the distributive law into account. This
        method calls multiply() repeatedly, leading to the code you will
        need to write.
        """
        factors = []
        for factor in self:
            if isinstance(factor, Product):
                factors += list(factor)
            else:
                factors.append(factor)
        result = Product([1])
        for factor in factors:
            result = multiply(result, simplify_if_possible(factor))
        return result.flatten()

    def flatten(self):
        """
        Simplifies nested products.
        """
        factors = []
        for factor in self:
            if isinstance(factor, Product):
                factors += list(factor)
            else:
                factors.append(factor)
        return Product(factors)

def simplify_if_possible(expr):
    """
    A helper function that guards against trying to simplify a non-Expression.
    """
    if isinstance(expr, Expression):
        return expr.simplify()
    else:
        return expr

# You may find the following helper functions to be useful.
# "multiply" is provided for you; but you will need to write "do_multiply"
# if you would like to use it.

def multiply(expr1, expr2):
    """
    This function makes sure that its arguments are represented as either a
    Sum or a Product, and then passes the hard work onto do_multiply.
    """
    # Simple expressions that are not sums or products can be handled
    # in exactly the same way as products -- they just have one thing in them.
    if not isinstance(expr1, Expression): expr1 = Product([expr1])
    if not isinstance(expr2, Expression): expr2 = Product([expr2])
    return do_multiply(expr1, expr2)


def do_multiply(expr1, expr2):
    """
    You have two Expressions, and you need to make a simplified expression
    representing their product. They are guaranteed to be of type Expression
    -- that is, either Sums or Products -- by the multiply() function that
    calls this one. Our classes are: expr1 is a Sum, and expr2 is a Sum;
    expr1 is a Sum, and expr2 is a Product; expr1 is a Product, and expr2 is a Sum;
    expr1 is a Product, and expr2 is a Product. We create Sums or Products 
    that represent what we get by applying the algebraic rules of 
    multiplication to these expressions, and simplifying.
    """
    # Replace this with your solution.
    raise NotImplementedError

