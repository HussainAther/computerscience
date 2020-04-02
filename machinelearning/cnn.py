import keras
import keras_utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import confusion_matrix, accuracy_score

"""
Convolutional neural network (CNN) in CIFAR-10 dataset
"""

# Load data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]

# Normalize inputs.
x_train2 = x_train / 255. - 0.5
x_test2 = x_test / 255. - 0.5

# Convert class labels to one-hot encoded, should have shape (?, NUM_CLASSES).
y_train2 = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test2 = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

def make_model():
    """
    Returns `Sequential` model.
    """
    model = Sequential()

    # CONV 1
    # first layer needs to define "input_shape"
    model.add(Conv2D(16, (3, 3), strides = (1, 1), padding="same", name = "conv1", input_shape=(32, 32, 3)))   
    model.add(LeakyReLU(0.1))
    
    # CONV 2
    model.add(Conv2D(32, (3, 3), strides = (1, 1), padding="same", name = "conv2"))  
    model.add(LeakyReLU(0.1))
    
    # MaxPooling2D 1
    model.add(MaxPooling2D((2, 2), name="max_pool_1"))
    
    # Dropout
    model.add(Dropout(0.25, noise_shape=None, seed=0))
    
    # CONV 3
    model.add(Conv2D(32, (3, 3), strides = (1, 1), padding="same", name = "conv3")) 
    model.add(LeakyReLU(0.1))
    
    # CONV 4
    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding="same", name = "conv4"))  
    model.add(LeakyReLU(0.1))
    
    # MaxPooling2D 1
    model.add(MaxPooling2D((2, 2), name="max_pool_2"))
    
    # Dropout
    model.add(Dropout(0.25, noise_shape=None, seed=0))
    
    # Flatten
    model.add(Flatten())    
    
    # FC
    model.add(Dense(256, name="fc1"))
    model.add(Dropout(0.5, noise_shape=None, seed=0))
    
    # FC
    model.add(Dense(NUM_CLASSES))  # the last layer with neuron for each class    
    model.add(Activation("softmax"))  # output probabilities

    return model

K.clear_session() # Clear default graph.
model = make_model()
model.summary()

# Train.
INIT_LR = 5e-3  # initial learning rate
BATCH_SIZE = 32
EPOCHS = 10

K.clear_session() # clear default graph
model = make_model() # define our model

# Prepare model for fitting (loss, optimizer, etc).
model.compile(
    loss="categorical_crossentropy",  
    optimizer=keras.optimizers.adamax(lr=INIT_LR), # for SGD
    metrics=["accuracy"] # Report accuracy during training.
)

# scheduler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch

# callback for printing of actual learning rate used by optimizer
class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))

# Fit model.
model.fit(
    x_train2, y_train2,  # prepared data
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler), LrHistory(), keras_utils.TqdmProgressCallback()],
    validation_data=(x_test2, y_test2),
    shuffle=True,
    verbose=0
)

# Make test predictions.
y_pred_test = model.predict_proba(x_test2)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probas = np.max(y_pred_test, axis=1)

# Metrics
plt.figure(figsize=(7, 6))
plt.title("Confusion matrix", fontsize=16)
plt.imshow(confusion_matrix(y_test, y_pred_test_classes))
plt.xticks(np.arange(10), cifar10_classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), cifar10_classes, fontsize=12)
plt.colorbar()
plt.show()
print("Test accuracy:", accuracy_score(y_test, y_pred_test_classes))

# Find input images with maximum stimuli.
K.clear_session() 
K.set_learning_phase(0) 
model = make_model()
model.load_weights("weights.h5") # that were saved after model.fit

def find_maximum_stimuli(layer_name, is_conv, filter_index, model, iterations=20, step=1., verbose=True):
    
    def image_values_to_rgb(x):
        # Normalize x: center on 0 (np.mean(x_train2)), ensure std is 0.25 (np.std(x_train2))
        # so that it looks like a normalized image input for our network
        x = (x - np.mean(x_train2)) / np.std(x_train2)

        # Reverse normalization to RGB values: x = (x_norm + 0.5) * 255.
        x = (x + 0.5) * 255
    
        # Clip values to [0, 255] and convert to bytes.
        x = np.clip(x, 0, 255).astype('uint8')
        return x
    input_img = model.input
    img_width, img_height = input_img.shape.as_list()[1:3]
    # Find the layer output by name.
    layer_output = list(filter(lambda x: x.name == layer_name, model.layers))[0].output
    # Build a loss function that maximizes the activation
    # of the filter_index filter of the layer considered.
    if is_conv:
        # mean over feature map values for convolutional layer
        loss = K.mean(layer_output[:, :, :, filter_index])
    else:
        loss = K.mean(layer_output[:, filter_index])

    # Compute the gradient of the loss wrt input image
    grads = K.gradients(loss, input_img)[0]  # [0] because of the batch dimension!
    # Normalize gradient.
    grads = grads / (K.sqrt(K.sum(K.square(grads))) + 1e-10)
    iterate = K.function([input_img], [loss, grads])
    # We start from a gray image with some random noise.
    input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * (0.1 if is_conv else 0.001)
    for i in range(iterations): # gradient ascent
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        if verbose:
            print('Current loss value:', loss_value)

    # decode the resulting input image
    img = image_values_to_rgb(input_img_data[0])
    
    return img, loss_value

# sample maximum stimuli
def plot_filters_stimuli(layer_name, is_conv, model, iterations=20, step=1., verbose=False):
    cols = 8
    rows = 2
    filter_index = 0
    max_filter_index = list(filter(lambda x: x.name == layer_name, model.layers))[0].output.shape.as_list()[-1] - 1
    fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
    for i in range(cols):
        for j in range(rows):
            if filter_index <= max_filter_index:
                ax = fig.add_subplot(rows, cols, i * rows + j + 1)
                ax.grid('off')
                ax.axis('off')
                loss = -1e20
                while loss < 0 and filter_index <= max_filter_index:
                    stimuli, loss = find_maximum_stimuli(layer_name, is_conv, filter_index, model,
                                                         iterations, step, verbose=verbose)
                    filter_index += 1
                if loss > 0:
                    ax.imshow(stimuli)
                    ax.set_title("Filter #{}".format(filter_index))
    plt.show()

# maximum stimuli for convolutional neurons
conv_activation_layers = []
for layer in model.layers:
    if isinstance(layer, LeakyReLU):
        prev_layer = layer.inbound_nodes[0].inbound_layers[0]
        if isinstance(prev_layer, Conv2D):
            conv_activation_layers.append(layer)

for layer in conv_activation_layers:
    print(layer.name)
    plot_filters_stimuli(layer_name=layer.name, is_conv=True, model=model)
