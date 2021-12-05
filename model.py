import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
from resnet import ResnetBlock


class Model(tf.keras.Model):


    def __init__(self):
        super(Model, self).__init__()

        # put params here
        self.dropout_rate = 0.3 # this is what we did in cnn
        self.drop_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.num_classes = 2 # start with this i guess

        self.learning_rate = 0.001 # from paper
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.batch_size= 64

        # making resnet variables

       # self.num_examples=num_examples


    def call(self, inputs):

        """
        Runs a forward pass on an input batch of images.
        inputs: input tensors of batch size x 128x128x1

        """



def main():
    return


if __name__ == '__main__':
    main()
