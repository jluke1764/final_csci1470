"""
This is where we will train, given whichever model is requested. We will call functions
from the preprocessing step and the actual model here.
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
from model import Model


def main():
    print('Click clack moo')
    my_model = Model()
    test_tensor = tf.zeros((10, 128, 128, 1))
    logits = my_model.call(test_tensor)

    return

if __name__ == '__main__':
    main()
