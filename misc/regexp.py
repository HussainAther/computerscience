import re

"""
Regexp practice.
"""

# Insert spaces between words with capital letters.
re.sub(r"(\w)([A-Z])", r"\1 \2", str1)
