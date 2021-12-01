"""
This is where we will train, given whichever model is requested. We will call functions
from the preprocessing step and the actual model here.
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
