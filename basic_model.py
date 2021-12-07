import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPool2D, Dropout, BatchNormalization


class BasicModel(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(BasicModel, self).__init__()

        self.batch_size = 100
        self.num_inputs = self.batch_size
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # Initialize all hyperparameters

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        #convolution layers
        self.convLayers = Sequential()
        self.convLayers.add(Conv2D(filters= 2, kernel_size=32, strides= [7,7], padding='same', activation='relu'))
        self.convLayers.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        self.convLayers.add(BatchNormalization(epsilon=.00005))

        self.convLayers.add(Conv2D(filters= 2, kernel_size= 64, strides= [5,5], padding='same', activation='relu'))
        self.convLayers.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        self.convLayers.add(BatchNormalization(epsilon=.00005))

        self.convLayers.add(Conv2D(filters= 2, kernel_size= 64, strides= [3,3], padding='same', activation='relu'))
        self.convLayers.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        self.convLayers.add(BatchNormalization(epsilon=.00005))

        #dense layer
        self.denseLayers = Sequential()
        self.denseLayers.add(Dense(64, activation='relu'))
        self.denseLayers.add(Dropout(rate=0.3))
        self.denseLayers.add(Dense(self.num_classes))




    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        conv_output = self.convLayers(inputs)
        logits = self.denseLayers(conv_output)



        return tf.reshape(logits, (self.batch_size, self.num_classes))


    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1, name=None)
        mean_loss = tf.reduce_mean(loss_tensor)
        return mean_loss

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """

        print("logit shapes", logits.shape)
        print("label shapes", labels.shape)
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))