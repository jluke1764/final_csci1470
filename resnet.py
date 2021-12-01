import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
import numpy as np


class ResnetBlock(tf.keras.Model):

     def __init__(self,filters,kernel_size, strides=1): # set default of strides to 1
         # filters, kernel_size, strides=(1, 1), padding='valid',
         self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='SAME') # supposed to be 3x3 but idk what is supposed to be 3x3
         self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size,strides=strides, padding='SAME') # do we need to have this?
         self.residual_conv = tf.keras.layers.Conv2D(filters,(1,1))

         self.batch_n_1 = tf.keras.layers.BatchNormalization() # this is learned
         self.batch_n_2 = tf.keras.layers.BatchNormalization() # this is learned

         self.relu = tf.keras.layers.Activation('relu')



    def call(self, inputs):
        original = inputs #idk if this properly saves the inputs in a fresh copy

        out_conv1 = self.conv1(inputs)
        out_batch1 = self.batch_n_1(out_conv1)
        relu_batch1 = self.relu(out_batch1)

        out_conv2 = self.conv2(relu_batch1)
        out_batch2 = self.batch_n_2(out_conv2)
        relu_batch2 = self.relu(out_batch2)

        # now we need to add the original back to this, but also gotta do a 1x1 convolution on it
        one_convolution = self.residual_conv(original)
        # should I do batch normalization here?

        combined = tf.keras.layers.Add()([relu_batch2, one_convolution])

        return self.relu(combined) # relu the addition too
