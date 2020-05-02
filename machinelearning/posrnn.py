import keras
import keras.layers as L
import nltk
import numpy as np
import sys

from collections import Counter, defaultdict
from keras.utils.np_utils import to_categorical
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

# Build model.
model = keras.models.Sequential()
model.add(L.InputLayer([None],dtype="int32"))
model.add(L.Embedding(len(all_words),50))
model.add(L.SimpleRNN(64,return_sequences=True))

# Add top layer that predicts tag probabilities.
stepwise_dense = L.Dense(len(all_tags),activation="softmax")
stepwise_dense = L.TimeDistributed(stepwise_dense)
model.add(stepwise_dense)

BATCH_SIZE=32
def generate_batches(sentences, batch_size=BATCH_SIZE, max_len=None, pad=0):
    while True:
        indices = np.random.permutation(np.arange(len(sentences)))
        for start in range(0,len(indices)-1,batch_size):
            batch_indices = indices[start:start+batch_size]
            batch_words,batch_tags = [],[]
            for sent in sentences[batch_indices]:
                words,tags = zip(*sent)
                batch_words.append(words)
                batch_tags.append(tags)

            batch_words = to_matrix(batch_words,word_to_id,max_len,pad)
            batch_tags = to_matrix(batch_tags,tag_to_id,max_len,pad)

            batch_tags_1hot = to_categorical(batch_tags,len(all_tags)).reshape(batch_tags.shape+(-1,))
            yield batch_words,batch_tags_1hot

def compute_test_accuracy(model):
    test_words,test_tags = zip(*[zip(*sentence) for sentence in test_data])
    test_words,test_tags = to_matrix(test_words,word_to_id),to_matrix(test_tags,tag_to_id)

    # Predict tag probabilities of shape [batch, time, n_tags].
    predicted_tag_probabilities = model.predict(test_words,verbose=1)
    predicted_tags = predicted_tag_probabilities.argmax(axis=-1)

    # Compute accurary excluding padding.
    numerator = np.sum(np.logical_and((predicted_tags == test_tags),(test_words != 0)))
    denominator = np.sum(test_words != 0)
    return float(numerator)/denominator


class EvaluateAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        sys.stdout.flush()
        print("\nMeasuring validation accuracy...")
        acc = compute_test_accuracy(self.model)
        print("\nValidation accuracy: %.5f\n"%acc)
        sys.stdout.flush()
        
# Compile and fit.
model.compile("adam","categorical_crossentropy")
model.fit_generator(generate_batches(train_data),len(train_data)/BATCH_SIZE,
                    callbacks=[EvaluateAccuracy()], epochs=5,)

# Test accuracy.
acc = compute_test_accuracy(model)
print("Final accuracy: %.5f"%acc)

# Define a model that utilizes bidirectional SimpleRNN.
model = keras.models.Sequential()

model.add(L.InputLayer([None],dtype="int32"))
model.add(L.Embedding(len(all_words),50))
model.add(L.Bidirectional(L.SimpleRNN(64,return_sequences=True)))

# Add top layer that predicts tag probabilities.
stepwise_dense = L.Dense(len(all_tags),activation="softmax")
stepwise_dense = L.TimeDistributed(stepwise_dense)
model.add(stepwise_dense)
