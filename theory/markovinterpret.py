import re

"""
A Markov interpreter. markov

In theoretical computer science, a Markov algorithm is a string rewriting system that uses 
grammar-like rules to operate on strings of symbols. Markov algorithms have been shown to be 
Turing-complete, which means that they are suitable as a general model of computation and can 
represent any mathematical expression from its simple notation. 

The example uses a regexp to parse the syntax of the grammar. This regexp is 
multi-line and verbose, and uses named groups to aid in understanding the regexp 
and to allow more meaningful group names to be used when extracting the replacement 
data from the grammars in function extractreplacements.

The example gains flexibility by not being tied to specific files. The functions 
may be imported into other programs which then can provide textual input from their 
sources without the need to pass "file handles" around.
"""

def extractreplacements(grammar):
    """
    For re syntax, extract the replacements.
    """
    return [ (matchobj.group("pat"), matchobj.group("repl"), bool(matchobj.group("term")))
                for matchobj in re.finditer(syntaxre, grammar)
                if matchobj.group('rule')]
 
def replace(text, replacements):
    """
    Replace text with replacements.
    """
    while True:
        for pat, repl, term in replacements:
            if pat in text:
                text = text.replace(pat, repl, 1)
                if term:
                    return text
                break
        else:
            return text

syntaxre = r"""(?mx)
^(?: 
  (?: (?P<comment> \# .* ) ) |
  (?: (?P<blank>   \s*  ) (?: \n | $ )  ) |
  (?: (?P<rule>    (?P<pat> .+? ) \s+ -> \s+ (?P<term> \.)? (?P<repl> .+) ) )
)$
""" 

# This rules file is extracted from Wikipedia:
# http://en.wikipedia.org/wiki/Markov_Algorithm
grammar1 = """\
A -> apple
B -> bag
S -> shop
T -> the
the shop -> my brother
a never used -> .terminating rule
"""
