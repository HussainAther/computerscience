import download_utils
import keras, keras.layers as L, keras.backend as K
import keras_utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lfw_dataset import load_lfw_dataset
from sklearn.model_selection import train_test_split

"""
Denoising autoencoders with principal component analysis (PCA)
"""

# Load data.
download_utils.link_week_4_resources()
X, attr = load_lfw_dataset(use_raw=True, dimx=32, dimy=32)
IMG_SHAPE = X.shape[1:]

# Center images.
X = X.astype("float32") / 255.0 - 0.5

# Split.
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

# Plot.
def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))

plt.title("sample images")

for i in range(6):
    plt.subplot(2,3,i+1)
    show_image(X[i])

print("X shape:", X.shape)
print("attr shape:", attr.shape)

# try to free memory
del X
import gc
gc.collect()

def build_pca_autoencoder(img_shape, code_size):
    """
    Here we define a simple linear autoencoder as described above for PCA.
    We also flatten and un-flatten data to be compatible with image shapes
    """
    
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Flatten()) # flatten image to vector
    encoder.add(L.Dense(code_size)) # actual encoder

    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(np.prod(img_shape))) # actual decoder, height*width*3 units
    decoder.add(L.Reshape(img_shape)) # un-flatten
    
    return encoder,decoder

s = reset_tf_session()

encoder, decoder = build_pca_autoencoder(IMG_SHAPE, code_size=32)

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

autoencoder.fit(x=X_train, y=X_train, epochs=15,
                validation_data=[X_test, X_test],
                callbacks=[keras_utils.TqdmProgressCallback()],
                verbose=0)
