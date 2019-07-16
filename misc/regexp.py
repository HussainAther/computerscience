import re

"""
Regexp practice.
"""

# Insert spaces between words with capital letters.
re.sub(r"(\w)([A-Z])", r"\1 \2", str1)

# Remove parenthesis area in a string.
re.sub(r" ?\([^)]+\)", "", str1)

# Split by delimiters.
re.split('; |, |\*|\n',text)
