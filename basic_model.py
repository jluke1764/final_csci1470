import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPool2D, Dropout, BatchNormalization


class BasicModel(tf.keras.Model):
    def __init__(self, num_classes):
        """
        This model class contains the architecture for your CNN that
        classifies images. It contains 3 convolution layers and 2 dense layers.
        """
        super(BasicModel, self).__init__()

        # hyperparameters
        self.batch_size = 64
        self.num_classes = num_classes # depends on input
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        #convolution layers
        self.convLayers = Sequential()
        self.convLayers.add(Conv2D(filters= 16, kernel_size=5, strides= [1,1], padding='same', activation='relu'))
        self.convLayers.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        self.convLayers.add(BatchNormalization(epsilon=.00005))

        self.convLayers.add(Conv2D(filters= 20, kernel_size= 5, strides= [1,1], padding='same', activation='relu'))
        self.convLayers.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        self.convLayers.add(BatchNormalization(epsilon=.00005))

        self.convLayers.add(Conv2D(filters= 20, kernel_size= 3, strides= [1,1], padding='same', activation='relu'))
        self.convLayers.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        self.convLayers.add(BatchNormalization(epsilon=.00005))
        self.convLayers.add(Flatten())

        # dense layers
        self.denseLayers = Sequential()
        self.denseLayers.add(Dense(32, activation='relu'))
        self.denseLayers.add(Dropout(rate=0.3))
        self.denseLayers.add(Dense(self.num_classes))


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images. Adapted from CSCI1470 HW2.

        :param inputs: images, shape of (num_inputs, 128, 128, 1); during training num_inputs is batch_size
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, num_classes)
        """
        conv_output = self.convLayers(inputs)
        logits = self.denseLayers(conv_output)
        return logits


    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function. Adapted from CSCI1470 HW2.

        :param logits: a matrix of shape (num_inputs, self.num_classes)
        containing the result of multiple convolution and feed forward layers. during training num_inputs is batch_size
        :param labels: matrix of shape (num_inputs, self.num_classes) containing the labels. during training num_inputs is batch_size.
        :return: the loss of the model as a Tensor
        """
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1, name=None)
        mean_loss = tf.reduce_mean(loss_tensor)
        return mean_loss

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels. Adapted from CSCI1470 HW2.

        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
