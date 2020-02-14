# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### One big advantage of autoencoders is that they can handle large datasets, with many instances and many features. So one strategy is to use an autoencoder to reduce the dimensionality down to a reasonable level, then use another dimensionality reduction algorithm for visualization.

# +
import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import requests

from collections import Counter

from PIL import Image
from io import BytesIO

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from matplotlib import cm
# %matplotlib inline

from keras import backend as K


from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# +
import numpy as np
import pandas as pd

X_train = pd.read_csv("A_X_train_processed.csv")

y_train = pd.read_csv('A_y_train.csv', header = None, index_col = 0, squeeze=True)
# -

X_train = X_train.drop('Unnamed: 0', axis=1)

X_train.shape

# ### building autoencoder

# +
### parameters of the simple autoencoder
input_dim = 19
encoder_dense_layers = [100, 30]
z_dim = 10
decoder_dense_layers = [30, 100]

### building the encoder
encoder_input = Input(shape=(input_dim,), name='encoder_input')
x = encoder_input
#x = Flatten()(x)

for i in range(len(encoder_dense_layers)):
    encoder_dense = Dense(encoder_dense_layers[i], activation='selu')
    x = encoder_dense(x)

encoder_output = Dense(z_dim)(x)
encoder = Model(encoder_input, encoder_output)

### building the decoder
decoder_input = Input(shape=(z_dim,), name = 'decoder_input')
x = decoder_input 

for i in range(len(decoder_dense_layers)):
    decoder_dense = Dense(decoder_dense_layers[i], activation = 'selu')
    x = decoder_dense(x)
    
x = Dense(input_dim)(x)
x = Activation('sigmoid')(x)
decoder_output = x
decoder = Model(decoder_input, decoder_output)
    
### full autoencoder
model_input = encoder_input
model_output = decoder(encoder_output)

autoencoder = Model(model_input, model_output)
# -

encoder.summary()

decoder.summary()

# ### running and compiling the autoencoder

autoencoder.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=['accuracy'])

history = autoencoder.fit(X_train, X_train, epochs=10000)


# ### let's look at the history plots

def history_plot(history, n):
    tot_pairs = np.int(len(list(history.history.keys()))/2)
    
    plt.figure(figsize=(18,12))
    for i in range(tot_pairs):

        plt.subplot(np.int(tot_pairs)*100+11+np.int(i))
        plt.plot(history.history[list(history.history.keys())[i]][n:],
                                     color='#4876ff', label = list(history.history.keys())[i])
        plt.plot(history.history[list(history.history.keys())[i + tot_pairs]][n:],
                                     color='slategray', label = list(history.history.keys())[i + tot_pairs])
        plt.xlabel("Epochs")
        plt.ylabel(list(history.history.keys())[i+tot_pairs])
        plt.legend()
        plt.grid()
        plt.title(list(history.history.keys())[i+tot_pairs])

    plt.show()


history_plot(history, 0)



# ### Now that we have trained a stacked autoencoder, we can use it to reduce the dataset’s dimensionality. For visualization, this does not give great results compared to other dimensionality reduction algorithms (such as those we discussed in Chapter 8), but one big advantage of autoencoders is that they can handle large datasets, with many instances and many features. So one strategy is to use an autoencoder to reduce the dimensionality down to a reasonable level, then use another dimensionality reduction algorithm for visualization. First, we use the encoder from our stacked autoencoder to reduce the dimensionality down to 30, then we use Scikit-Learn’s implementation of the t-SNE algorithm to reduce the dimensionality down to 2 for visualization:

# ### autoencoder + TSNE

X_train_compressed = encoder.predict(X_train)
X_train_compressed.shape

# +
from sklearn.manifold import TSNE

tsne = TSNE(n_components = 2, random_state = 0)
X_train_2D = tsne.fit_transform(X_train_compressed)

# +
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=y_train, s=10, cmap="tab10")

for i, txt in enumerate(y_train[:500]):
    plt.annotate(txt, (X_train_2D[i, 0], X_train_2D[i, 1]))
# -

# ### autoencoder + PCA

from sklearn.decomposition import PCA 
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X_train_compressed)

pca.explained_variance_ratio_

# +
figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
colors = ['red', 'green']
plt.scatter(X2D[:, 0], X2D[:, 1], c=y_train, zorder=10, s=2, cmap=matplotlib.colors.ListedColormap(colors))

for i, txt in enumerate(y_train):
    plt.annotate(txt, (X2D[i, 0], X2D[i, 1]))
# -

pca = PCA()
pca.fit(X_train_compressed)
cumsum = np.cumsum(pca.explained_variance_ratio_) 
d = np.argmax(cumsum >= 0.95) + 1

d


