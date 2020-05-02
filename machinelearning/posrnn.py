import nltk
import numpy as np
import sys

from collections import Counter
from sklearn.cross_validation import train_test_split

# Parts of speech (POS) recurrent neural networks with keras

"""
ADJ - adjective (new, good, high, ...)
ADP - adposition (on, of, at, ...)
ADV - adverb (really, already, still, ...)
CONJ - conjunction (and, or, but, ...)
DET - determiner, article (the, a, some, ...)
NOUN - noun (year, home, costs, ...)
NUM - numeral (twenty-four, fourth, 1991, ...)
PRT - particle (at, on, out, ...)
PRON - pronoun (he, their, her, ...)
VERB - verb (is, say, told, ...)
. - punctuation marks (. , ;)
X - other (ersatz, esprit, dunno, ...)
"""

# Download and process data.
nltk.download("brown")
nltk.download("universal_tagset")
data = nltk.corpus.brown.tagged_sents(tagset="universal")
all_tags = ["#EOS#","#UNK#","ADV", "NOUN", "ADP", "PRON", "DET", ".", "PRT", "VERB", "X", "NUM", "CONJ", "ADJ"]
data = np.array([ [(word.lower(),tag) for word,tag in sentence] for sentence in data ])

# Split between training and testing data.
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)


# Build vocabulary.
word_counts = Counter()
for sentence in data:
    words, tags = zip(*sentence)
    word_counts.update(words)
