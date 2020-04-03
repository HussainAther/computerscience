import download_utils
import keras, keras.layers as L, keras.backend as K
import keras_utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lfw_dataset import load_lfw_dataset
from sklearn.model_selection import train_test_split

"""
Creating autoencoders with principal component analysis (PCA)
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
autoencoder.compile(optimizer="adamax", loss="mse")

autoencoder.fit(x=X_train, y=X_train, epochs=15,
                validation_data=[X_test, X_test],
                callbacks=[keras_utils.TqdmProgressCallback()],
                verbose=0)

def visualize(img,encoder,decoder):
    """
    Draw original, encoded and decoded images.
    """
    code = encoder.predict(img[None])[0] # img[None] is the same as img[np.newaxis, :]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()

score = autoencoder.evaluate(X_test,X_test,verbose=0)
print("PCA MSE:", score)

for i in range(5):
    img = X_test[i]
    visualize(img,encoder,decoder)

"""
Convolutional autoencoder
"""

def test_conv2d_transpose(img_size, filter_size):
    """
    Transpose convolution for the images
    """
    print("Transpose convolution test for img_size={}, filter_size={}:".format(img_size, filter_size))
    x = (np.arange(img_size ** 2, dtype=np.float32) + 1).reshape((1, img_size, img_size, 1))
    f = (np.ones(filter_size ** 2, dtype=np.float32)).reshape((filter_size, filter_size, 1, 1))
    conv = tf.nn.conv2d_transpose(x, f, 
                                  output_shape=(1, img_size * 2, img_size * 2, 1), 
                                  strides=[1, 2, 2, 1], 
                                  padding="SAME")
    with tf.Session() as session:
        result = session.run(conv)
        print("input:")
        print(x[0, :, :, 0])
        print("filter:")
        print(f[:, :, 0, 0])
        print("output:")
        print(result[0, :, :, 0])
        
test_conv2d_transpose(img_size=2, filter_size=2)

def build_deep_autoencoder(img_shape, code_size):
    """
    Deep autoencoding 
    """
    H,W,C = img_shape
    
    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Conv2D(32, (3, 3), strides = (1, 1), padding="same", activation="elu"))
    encoder.add(L.MaxPooling2D((2, 2)))
    encoder.add(L.Conv2D(64, (3, 3), strides = (1, 1), padding="same", activation="elu"))
    encoder.add(L.MaxPooling2D((2, 2)))
    encoder.add(L.Conv2D(128, (3, 3), strides = (1, 1), padding="same", activation="elu"))
    encoder.add(L.MaxPooling2D((2, 2)))
    encoder.add(L.Conv2D(256, (3, 3), strides = (1, 1), padding="same", activation="elu"))
    encoder.add(L.MaxPooling2D((2, 2)))
    encoder.add(L.Flatten()) # flatten image to vector
    encoder.add(L.Dense(code_size)) # actual encoder
    
    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(2*2*256))                 #actual encoder 
    decoder.add(L.Reshape((2,2,256)))         #un-flatten
    decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation="elu", padding="same"))
    decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation="elu", padding="same"))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation="elu", padding="same"))
    decoder.add(L.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, activation=None, padding="same"))
    
    return encoder, decoder

# Check autoencoder shapes along different code_sizes.
get_dim = lambda layer: np.prod(layer.output_shape[1:])
for code_size in [1,8,32,128,512]:
    s = reset_tf_session()
    encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=code_size)
    print("Testing code size %i" % code_size)
    assert encoder.output_shape[1:]==(code_size,),"encoder must output a code of required size"
    assert decoder.output_shape[1:]==IMG_SHAPE, "decoder must output an image of valid shape"
    assert len(encoder.trainable_weights)>=6, "encoder must contain at least 3 layers"
    assert len(decoder.trainable_weights)>=6, "decoder must contain at least 3 layers"
    for layer in encoder.layers + decoder.layers:
        assert get_dim(layer) >= code_size, "Encoder layer %s is smaller than bottleneck (%i units)"%(layer.name,get_dim(layer))

print("All tests passed!")

# Print summary report.
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)
encoder.summary()
decoder.summary()

# Build it.
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)
inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adamax", loss="mse")

# Fit.
autoencoder.fit(x=X_train, y=X_train, epochs=25,
                validation_data=[X_test, X_test],
                callbacks=[keras_utils.ModelSaveCallback(model_filename),
                           keras_utils.TqdmProgressCallback()],
                verbose=0,
                initial_epoch=last_finished_epoch or 0)

# See how well it worked.
reconstruction_mse = autoencoder.evaluate(X_test, X_test, verbose=0)
print("Convolutional autoencoder MSE:", reconstruction_mse)
for i in range(5):
    img = X_test[i]
    visualize(img,encoder,decoder)

# Denoising.
def apply_gaussian_noise(X,sigma=0.1):
    """
    Add noise from standard normal distribution with standard deviation sigma.
    :param X: image tensor of shape [batch,height,width,3]
    Returns X + noise.
    """
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

# noise tests
theoretical_std = (X_train[:100].std()**2 + 0.5**2)**.5
our_std = apply_gaussian_noise(X_train[:100],sigma=0.5).std()
assert abs(theoretical_std - our_std) < 0.01, "Standard deviation does not match it"s required value. Make sure you use sigma as std."
assert abs(apply_gaussian_noise(X_train[:100],sigma=0.5).mean() - X_train[:100].mean()) < 0.01, "Mean has changed. Please add zero-mean noise"

# Plot.
plt.subplot(1,4,1)
show_image(X_train[0])
plt.subplot(1,4,2)
show_image(apply_gaussian_noise(X_train[:1],sigma=0.01)[0])
plt.subplot(1,4,3)
show_image(apply_gaussian_noise(X_train[:1],sigma=0.1)[0])
plt.subplot(1,4,4)
show_image(apply_gaussian_noise(X_train[:1],sigma=0.5)[0])

# Train.
s = reset_tf_session()

# we use bigger code size here for better quality
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=512)
assert encoder.output_shape[1:]==(512,), "encoder must output a code of required size"

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inp, reconstruction)
autoencoder.compile("adamax", "mse")

for i in range(25):
    print("Epoch %i/25, Generating corrupted samples..."%(i+1))
    X_train_noise = apply_gaussian_noise(X_train)
    X_test_noise = apply_gaussian_noise(X_test)
    autoencoder.fit(x=X_train_noise, y=X_train, epochs=1,
                    validation_data=[X_test_noise, X_test],
                    callbacks=[keras_utils.TqdmProgressCallback()],
                    verbose=0)

X_test_noise = apply_gaussian_noise(X_test)
denoising_mse = autoencoder.evaluate(X_test_noise, X_test, verbose=0)
print("Denoising MSE:", denoising_mse)
for i in range(5):
    img = X_test_noise[i]
    visualize(img,encoder,decoder)
