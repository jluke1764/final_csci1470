import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *

#from resnet import ResnetBlock


class Model(tf.keras.Model):


    def __init__(self):
        super(Model, self).__init__()

        # put params here
        self.dropout_rate = 0.3 # this is what we did in cnn
        self.drop_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.num_classes = 10 # start with this i guess

        self.learning_rate = 0.01 # from paper 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.batch_size= 64

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


       # self.num_examples=num_examples

    @tf.function
    def call(self, inputs):

        """
        Runs a forward pass on an input batch of images.
        inputs: input tensors of batch size x 128x128x1
        outputs: batch size x num_classes

        """
        # add dropout first here
        init = self.initial_conv(inputs)
        init_normal = self.batch0(init)
        print('initial ', init_normal.shape)


        # first 64x64x64
        conv1a = self.conv1a(init_normal)
        batch1a = self.batch1a(conv1a)
        relu1a = tf.keras.layers.Activation('relu')(batch1a)
        conv2a = self.conv2a(relu1a)
        batch2a = self.batch2a(conv2a)
        resa = tf.keras.layers.Add()([init_normal, batch2a])
        resa = tf.keras.layers.Activation('relu')(resa)
        print('first 64x64x64 ', resa.shape)

        # second 64x64x64
        conv1b = self.conv1b(resa)
        batch1b = self.batch1b(conv1b)
        relu1b = tf.keras.layers.Activation('relu')(batch1b)
        conv2b = self.conv2b(relu1b)
        batch2b = self.batch2b(conv2b)
        resb = tf.keras.layers.Add()([resa, batch2b])
        resb = tf.keras.layers.Activation('relu')(resb)
        print('second 64x64x64 ', resb.shape)

        # third 64x64x64
        conv1c = self.conv1c(resb)
        batch1c = self.batch1c(conv1c)
        relu1c = tf.keras.layers.Activation('relu')(batch1c)
        conv2c = self.conv2c(relu1c)
        batch2c = self.batch2c(conv2c)
        resc = tf.keras.layers.Add()([resb, batch2c])
        resc = tf.keras.layers.Activation('relu')(resc)
        print('third 64x64x64 ', resc.shape)

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
        print('first 32x32x128 ', resd.shape)

        #second 32x32x128
        conv1e = self.conv1e(resd)
        batch1e = self.batch1e(conv1e)
        relu1e = tf.keras.layers.Activation('relu')(batch1e)
        conv2e = self.conv2e(relu1e)
        batch2e = self.batch2e(conv2e)
        rese = tf.keras.layers.Add()([resd, batch2e])
        rese = tf.keras.layers.Activation('relu')(rese)
        print('second 32x32x128 ', rese.shape)

        # third 32x32x128
        conv1F = self.conv1F(rese)
        batch1F = self.batch1F(conv1F)
        relu1F = tf.keras.layers.Activation('relu')(batch1F)
        conv2F = self.conv2F(relu1F)
        batch2F = self.batch2F(conv2F)
        resF = tf.keras.layers.Add()([rese, batch2F])
        resF = tf.keras.layers.Activation('relu')(resF)
        print('third 32x32x128 ', resF.shape)

        # dropout
        resF = self.drop_layer(resF)

        # FIRST 16x16x256
        conv1G = self.conv1G(resF)
        batch1G = self.batch1G(conv1G)
        relu1G = tf.keras.layers.Activation('relu')(batch1G)
        conv2G = self.conv2G(relu1G)
        batch2G = self.batch2G(conv2G)

        half_size = self.half_256(resF)
        half_size = self.pool256(half_size)


        resG = tf.keras.layers.Add()([half_size, batch2G])
        resG = tf.keras.layers.Activation('relu')(resG)
        print('first 16x16x256 ', resG.shape)


        #SECOND 32x32x128
        conv1H = self.conv1H(resG)
        batch1H = self.batch1H(conv1H)
        relu1H = tf.keras.layers.Activation('relu')(batch1H)
        conv2H = self.conv2H(relu1H)
        batch2H = self.batch2H(conv2H)
        resH = tf.keras.layers.Add()([resG, batch2H])
        resH = tf.keras.layers.Activation('relu')(resH)
        print('second 16x16x256 ', resH.shape)

        #THIRD 32x32x128
        conv1I = self.conv1I(resH)
        batch1I = self.batch1I(conv1I)
        relu1I = tf.keras.layers.Activation('relu')(batch1I)
        conv2I = self.conv2I(relu1I)
        batch2I = self.batch2I(conv2I)
        resI = tf.keras.layers.Add()([resH, batch2I])
        resI = tf.keras.layers.Activation('relu')(resI)
        print('last 16x16x256 ', resI.shape)

        # dropout
        resI = self.drop_layer(resI)

        pooled = self.pooling_layer_final(resI)
        pooled = self.drop_layer(pooled)

        res = self.dense_layer(pooled)
        res = tf.reshape(res, [-1,10])

        print(res.shape)

        return res




    def loss(self, logits, labels): # this is directly from the cnn assignment
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        sum = tf.math.reduce_sum(loss)
        mean_loss = tf.math.reduce_mean(sum)

        return mean_loss

    def accuracy(self, logits, labels): # from the cnn assignment
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))




def main():
    return


if __name__ == '__main__':
    main()
