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

# Find all adverbs and positions.
for m in re.finditer(r"\w+ly", text):
    print('%d-%d: %s' % (m.start(), m.end(), m.group(0)))
