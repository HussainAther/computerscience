import nltk
import numpy as np
import sys

from collections import Counter, defaultdict
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

all_words = ["#EOS#","#UNK#"]+list(list(zip(*word_counts.most_common(10000)))[0])

# Measure what fraction of data words are in the dictionary.
print("Coverage = %.5f"%(float(sum(word_counts[w] for w in all_words)) / sum(word_counts.values())))

word_to_id = defaultdict(lambda:1, {word:i for i,word in enumerate(all_words)})
tag_to_id = {tag:i for i,tag in enumerate(all_tags)}

def to_matrix(lines, token_to_id, max_len=None, pad=0, dtype="int32", time_major=False):
    """
    Converts a list of names into rnn-digestable matrix with paddings added after the end.
    """
    
    max_len = max_len or max(map(len,lines))
    matrix = np.empty([len(lines),max_len],dtype)
    matrix.fill(pad)

    for i in range(len(lines)):
        line_ix = list(map(token_to_id.__getitem__,lines[i]))[:max_len]
        matrix[i,:len(line_ix)] = line_ix

    return matrix.T if time_major else matrix

batch_words, batch_tags = zip(*[zip(*sentence) for sentence in data[-3:]])
