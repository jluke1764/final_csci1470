import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *

#from resnet import ResnetBlock


class ResnetModel(tf.keras.Model):
    """
    This model class contains the architecture for your CNN that
    classifies images. It contains 3 convolution layers and 2 dense layers.
    Model architecture adapted from Lu and Tran, Stanford 2017.
    """

    def __init__(self, num_classes):
        super(ResnetModel, self).__init__()
        # hyperparameters
        self.dropout_rate = 0.3
        self.drop_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.num_classes = num_classes

        self.learning_rate = 0.001 # from paper 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.batch_size = 64

        # layers for resnet
        self.initial_conv = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='SAME')
        self.batch0 = tf.keras.layers.BatchNormalization()

        self.conv1a = tf.keras.layers.Conv2D(64, 3,  padding='SAME')
        self.batch1a = tf.keras.layers.BatchNormalization()
        self.conv2a = tf.keras.layers.Conv2D(64, 3,  padding='SAME')
        self.batch2a = tf.keras.layers.BatchNormalization()

        self.conv1b = tf.keras.layers.Conv2D(64, 3,  padding='SAME')
        self.batch1b = tf.keras.layers.BatchNormalization()
        self.conv2b = tf.keras.layers.Conv2D(64, 3,  padding='SAME')
        self.batch2b = tf.keras.layers.BatchNormalization()

        self.conv1c = tf.keras.layers.Conv2D(64, 3,  padding='SAME')
        self.batch1c = tf.keras.layers.BatchNormalization()
        self.conv2c = tf.keras.layers.Conv2D(64, 3,  padding='SAME')
        self.batch2c = tf.keras.layers.BatchNormalization()

        self.drop_layer = tf.keras.layers.Dropout(self.dropout_rate)

        self.half_128 = tf.keras.layers.Conv2D(128, 1, padding='SAME')
        self.pool128 = tf.keras.layers.AveragePooling2D(strides=(2,2), pool_size=(2, 2))

        self.conv1d = tf.keras.layers.Conv2D(128, 3,  padding='SAME', strides=2)
        self.batch1d = tf.keras.layers.BatchNormalization()
        self.conv2d = tf.keras.layers.Conv2D(128, 3,  padding='SAME', strides=1)
        self.batch2d = tf.keras.layers.BatchNormalization()

        self.conv1e = tf.keras.layers.Conv2D(128, 3,  padding='SAME', strides=1)
        self.batch1e = tf.keras.layers.BatchNormalization()
        self.conv2e = tf.keras.layers.Conv2D(128, 3,  padding='SAME', strides=1)
        self.batch2e = tf.keras.layers.BatchNormalization()

        self.conv1F= tf.keras.layers.Conv2D(128, 3,  padding='SAME', strides=1)
        self.batch1F = tf.keras.layers.BatchNormalization()
        self.conv2F = tf.keras.layers.Conv2D(128, 3,  padding='SAME', strides=1)
        self.batch2F = tf.keras.layers.BatchNormalization()

        self.half_256 = tf.keras.layers.Conv2D(256, 1, padding='SAME')
        self.pool256 = tf.keras.layers.AveragePooling2D(strides=(2,2), pool_size=(2, 2))
        self.conv1G= tf.keras.layers.Conv2D(256, 3,  padding='SAME', strides=2)
        self.batch1G = tf.keras.layers.BatchNormalization()
        self.conv2G = tf.keras.layers.Conv2D(256, 3,  padding='SAME', strides=1)
        self.batch2G = tf.keras.layers.BatchNormalization()

        self.conv1H= tf.keras.layers.Conv2D(256, 3,  padding='SAME', strides=1)
        self.batch1H = tf.keras.layers.BatchNormalization()
        self.conv2H = tf.keras.layers.Conv2D(256, 3,  padding='SAME', strides=1)
        self.batch2H = tf.keras.layers.BatchNormalization()

        self.conv1I= tf.keras.layers.Conv2D(256, 3,  padding='SAME', strides=1)
        self.batch1I = tf.keras.layers.BatchNormalization()
        self.conv2I = tf.keras.layers.Conv2D(256, 3,  padding='SAME', strides=1)
        self.batch2I = tf.keras.layers.BatchNormalization()

        self.pooling_layer_final = tf.keras.layers.AveragePooling2D(pool_size=(16, 16))
        self.dense_layer = tf.keras.layers.Dense(self.num_classes, input_shape=(512,))


    @tf.function
    def call(self, inputs):

        """
        Runs a forward pass on an input batch of images.
        inputs: input tensors of batch sizex128x128x1
        outputs: tensor of batch size x num_classes

        """
        init = self.initial_conv(inputs)
        init_normal = self.batch0(init)

        # first 64x64x64
        conv1a = self.conv1a(init_normal) # first convolution
        batch1a = self.batch1a(conv1a)
        relu1a = tf.keras.layers.Activation('relu')(batch1a)
        conv2a = self.conv2a(relu1a) # second convolution
        batch2a = self.batch2a(conv2a)
        resa = tf.keras.layers.Add()([init_normal, batch2a]) # add original input with the res of the two convolutions
        resa = tf.keras.layers.Activation('relu')(resa)
        # second 64x64x64
        conv1b = self.conv1b(resa)
        batch1b = self.batch1b(conv1b)
        relu1b = tf.keras.layers.Activation('relu')(batch1b)
        conv2b = self.conv2b(relu1b)
        batch2b = self.batch2b(conv2b)
        resb = tf.keras.layers.Add()([resa, batch2b])
        resb = tf.keras.layers.Activation('relu')(resb)
        # third 64x64x64
        conv1c = self.conv1c(resb)
        batch1c = self.batch1c(conv1c)
        relu1c = tf.keras.layers.Activation('relu')(batch1c)
        conv2c = self.conv2c(relu1c)
        batch2c = self.batch2c(conv2c)
        resc = tf.keras.layers.Add()([resb, batch2c])
        resc = tf.keras.layers.Activation('relu')(resc)
        # dropout
        resc = self.drop_layer(resc)

        #first 32x32x128
        conv1d = self.conv1d(resc)
        batch1d = self.batch1d(conv1d)
        relu1d = tf.keras.layers.Activation('relu')(batch1d)
        conv2d = self.conv2d(relu1d)
        batch2d = self.batch2d(conv2d)
        half_size = self.half_128(resc)
        half_size = self.pool128(half_size)
        resd = tf.keras.layers.Add()([half_size, batch2d])
        resd = tf.keras.layers.Activation('relu')(resd)
        #second 32x32x128
        conv1e = self.conv1e(resd)
        batch1e = self.batch1e(conv1e)
        relu1e = tf.keras.layers.Activation('relu')(batch1e)
        conv2e = self.conv2e(relu1e)
        batch2e = self.batch2e(conv2e)
        rese = tf.keras.layers.Add()([resd, batch2e])
        rese = tf.keras.layers.Activation('relu')(rese)
        # third 32x32x128
        conv1F = self.conv1F(rese)
        batch1F = self.batch1F(conv1F)
        relu1F = tf.keras.layers.Activation('relu')(batch1F)
        conv2F = self.conv2F(relu1F)
        batch2F = self.batch2F(conv2F)
        resF = tf.keras.layers.Add()([rese, batch2F])
        resF = tf.keras.layers.Activation('relu')(resF)
        # dropout
        resF = self.drop_layer(resF)

        # first 16x16x256
        conv1G = self.conv1G(resF)
        batch1G = self.batch1G(conv1G)
        relu1G = tf.keras.layers.Activation('relu')(batch1G)
        conv2G = self.conv2G(relu1G)
        batch2G = self.batch2G(conv2G)
        half_size = self.half_256(resF)
        half_size = self.pool256(half_size)
        resG = tf.keras.layers.Add()([half_size, batch2G])
        resG = tf.keras.layers.Activation('relu')(resG)
        # second 16x16x256
        conv1H = self.conv1H(resG)
        batch1H = self.batch1H(conv1H)
        relu1H = tf.keras.layers.Activation('relu')(batch1H)
        conv2H = self.conv2H(relu1H)
        batch2H = self.batch2H(conv2H)
        resH = tf.keras.layers.Add()([resG, batch2H])
        resH = tf.keras.layers.Activation('relu')(resH)
        #third 16x16x256
        conv1I = self.conv1I(resH)
        batch1I = self.batch1I(conv1I)
        relu1I = tf.keras.layers.Activation('relu')(batch1I)
        conv2I = self.conv2I(relu1I)
        batch2I = self.batch2I(conv2I)
        resI = tf.keras.layers.Add()([resH, batch2I])
        resI = tf.keras.layers.Activation('relu')(resI)
        # dropout
        resI = self.drop_layer(resI)

        #pooling, dropout
        pooled = self.pooling_layer_final(resI)
        pooled = self.drop_layer(pooled)

        # flatten before dense
        output = tf.keras.layers.Flatten()(pooled)
        res = self.dense_layer(output)
        return res


    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function. Adapted from HW2.
        :param logits: a matrix of shape (num_inputs, self.num_classes)
        containing the result of multiple convolution and feed forward layers during training num_inputs is batch_size
        :param labels: matrix of shape (batch_size, self.num_classes) containing the labels
        :return: the loss of the model as a Tensor
        """
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        sum = tf.math.reduce_sum(loss)
        mean_loss = tf.math.reduce_mean(sum)

        return mean_loss

    def accuracy(self, logits, labels): # from the cnn assignment
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels.  Adapted from HW2.
        :param logits: a matrix of size (num_inputs, self.num_classes);
        containing the result of multiple convolution and feed forward layers during training num_inputs is batch_size
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))



