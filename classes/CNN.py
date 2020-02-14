import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from utils.callbacks import CustomCallback, step_decay_schedule

import numpy as np
import json
import os
import pickle

import matplotlib.pyplot as plt

class CNN():
    def __init__(self
        , model_name
        , input_dim
        , conv_filters
        , conv_kernel_size
        , conv_strides
        , dense_layers
        , z_dim
        , use_batch_norm = False
        , use_dropout = False
        ):

        self.model_name = model_name

        self.input_dim = input_dim
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.dense_layers = dense_layers
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers = len(conv_filters)

        self._build()

    def _build(self):

        input = Input(shape=self.input_dim, name='NN_input')

        x = input

        for i in range(self.n_layers):
            conv_layer = Conv2D(
                filters = self.conv_filters[i]
                , kernel_size = self.conv_kernel_size[i]
                , strides = self.conv_strides[i]
                , padding = 'same'
                , name = 'conv_' + str(i)
                )

            x = conv_layer(x)

            if i < self.n_layers - 1:
                x = LeakyReLU(0.02)(x)

                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                if self.use_dropout:
                    x = Dropout(rate = 0.25)(x)
            else:
### here we add the dense layers after the convolution layers before final layer
                x = Flatten()(x)
                for j in range(len(self.dense_layers)):
                    Dense(self.dense_layers[j], activation="relu")(x)

        self.shape_before_flattening = K.int_shape(x)[1:]

        x= Dense(self.z_dim, name='output')(x)
        output = Activation('softmax')(x)

        self.model = Model(input, output)


    def compile(self, learning_rate, metric, loss):
        self.learning_rate = learning_rate
        self.metric = metric
        self.loss = loss

        optimizer = Adam(lr=learning_rate)

        self.model.compile(optimizer=optimizer, loss = loss, metrics = [metric])

    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'models'))

        with open(os.path.join(folder, 'CNN_params.pkl'), 'wb') as f:
            pickle.dump([
                self.name
                ,self.input_dim
                ,self.conv_filters
                ,self.conv_kernel_size
                ,self.conv_strides
                ,self.dense_layers
                ,self.z_dim
                ,self.use_batch_norm
                ,self.use_dropout
                ], f)

        self.model.save(folder+'models/'+self.model_name+'.h5')
        plot_model(self.model, to_file=os.path.join(folder ,'viz/'+self.model_name+'.png'), show_shapes = True, show_layer_names = True)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, x_train, y_train,
                x_val, y_val, folder,
                batch_size, epochs, print_every_n_batches = 100,
                initial_epoch = 0, lr_decay = 1):

        #checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath = folder+"/weights/"+self.model_name+"_weights.h5", verbose = 2, save_best_only=True)
        #early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        #callback_list = [checkpoint_cb, early_stopping_cb]
        #checkpointer = ModelCheckpoint(filepath='model.h5', verbose=2, save_best_only=True)


        self.history = self.model.fit(
            x_train
            , y_train
            , validation_data = [x_val, y_val]
            , batch_size = batch_size
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
        #    , callbacks = checkpointer
        )


    def history_plot(self, n=0):
        plt.figure(figsize=(18, 12))

        plt.subplot(211)
        plt.plot(self.history.history['loss'][n:], color='slategray', label = 'train')
        plt.plot(self.history.history['val_loss'][n:], color='#4876ff', label = 'valid')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(self.loss)

        plt.subplot(212)
        plt.plot(self.history.history[self.metric][n:], color='slategray', label = 'train')
        plt.plot(self.history.history['val_'+self.metric][n:], color='#4876ff', label = 'valid')
        plt.xlabel("Epochs")
        plt.ylabel(self.metric)
        plt.legend()
        plt.title(self.metric)

        plt.show()
