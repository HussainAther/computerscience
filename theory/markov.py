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

# Slightly modified from the rules on Wikipedia
grammar2 = """\
A -> apple
B -> bag
S -> .shop
T -> the
the shop -> my brother
a never used -> .terminating rule
"""

# BNF (Backus-Naur form) Syntax testing rules backus naur Backus Naur
grammar3 = """\
A -> apple
WWWW -> with
Bgage -> ->.*
B -> bag
->.* -> money
W -> WW
S -> .shop
T -> the
the shop -> my brother
a never used -> .terminating rule
"""

### Unary Multiplication Engine, for testing Markov Algorithm implementation (unary multiplication)
grammar4 = """\
# Unary addition engine
_+1 -> _1+
1+1 -> 11+
# Pass for converting from the splitting of multiplication into ordinary
# addition
1! -> !1
,! -> !+
_! -> _
# Unary multiplication by duplicating left side, right side times
1*1 -> x,@y
1x -> xX
X, -> 1,1
X1 -> 1X
_x -> _X
,x -> ,X
y1 -> 1y
y_ -> _
# Next phase of applying
1@1 -> x,@y
1@_ -> @_
,@_ -> !_
++ -> +
# Termination cleanup for addition
_1 -> 1
1+_ -> 1
_+_ -> 
""" 

"""
Turing machine: three-state busy beaver used in the Busy Beaver game.

More precisely, the busy beaver game consists of designing a halting, 
binary-alphabet Turing machine which writes the most 1s on the tape, 
using only a given set of states. The rules for the 2-state game are as follows: 
The machine must have two states in addition to the halting state, and
the tape initially contains 0s only.
"""

grammar5 = """\
# state A, symbol 0 => write 1, move right, new state B
A0 -> 1B
# state A, symbol 1 => write 1, move left, new state C
0A1 -> C01
1A1 -> C11
# state B, symbol 0 => write 1, move left, new state A
0B0 -> A01
1B0 -> A11
# state B, symbol 1 => write 1, move right, new state B
B1 -> 1B
# state C, symbol 0 => write 1, move left, new state B
0C0 -> B01
1C0 -> B11
# state C, symbol 1 => write 1, move left, halt
0C1 -> H01
1C1 -> H11
"""

text1 = "I bought a B of As from T S."
 
text2 = "I bought a B of As W my Bgage from T S."
 
text3 = "_1111*11111_"
 
text4 = "000000A000000"

if __name__ == "__main__":
    assert replace(text1, extractreplacements(grammar1)) \
           == "I bought a bag of apples from my brother."
    assert replace(text1, extractreplacements(grammar2)) \
           == "I bought a bag of apples from T shop."
    # Stretch goals
    assert replace(text2, extractreplacements(grammar3)) \
           == "I bought a bag of apples with my money from T shop."
    assert replace(text3, extractreplacements(grammar4)) \
           == "11111111111111111111"
    assert replace(text4, extractreplacements(grammar5)) \
           == "00011H1111000"
