"""
A Markov interpreter.

The example uses a regexp to parse the syntax of the grammar. This regexp is 
multi-line and verbose, and uses named groups to aid in understanding the regexp 
and to allow more meaningful group names to be used when extracting the replacement 
data from the grammars in function extractreplacements.

The example gains flexibility by not being tied to specific files. The functions 
may be imported into other programs which then can provide textual input from their 
sources without the need to pass "file handles" around.
"""
