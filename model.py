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
        self.num_classes = 10 # start with this i guess

        self.learning_rate = 0.001 # from paper
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)


    def call(self, inputs):

        """
        Runs a forward pass on an input batch of images.
        inputs: input tensors of batch size x 128x128x1

        """
        print('input shape is ', inputs.shape)
        dropped_out = self.drop_layer(inputs)
        self.initial_conv = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='SAME') # idk params
        first_conv = self.initial_conv(inputs)
        print('after 7x7 conv ', first_conv.shape)


        resnet_1 = ResnetBlock(64,3) # filters, kernel_size, strides (optional, set to 1)
        res1 = resnet_1.call(first_conv)
        resnet_2 = ResnetBlock(64,3) # filters, kernel_size
        res2 = resnet_1.call(res1)
        resnet_3 = ResnetBlock(64,3) # filters, kernel_size
        res3 = resnet_1.call(res2)
        dropped_out_res3 = self.drop_layer(res3)
        print('after initial res blocks ', dropped_out_res3.shape)


        resnet_4 = ResnetBlock(128,3, strides=(2,2)) # filters, kernel_size, strides (optional, set to 1)
        res4 = resnet_4.call(dropped_out_res3)

        resnet_5 = ResnetBlock(128,3) # filters, kernel_size
        res5 = resnet_5.call(res4)
        resnet_6 = ResnetBlock(128,3) # filters, kernel_size
        res6 = resnet_6.call(res5)
        dropped_out_res6 = self.drop_layer(res6)

        resnet_7 = ResnetBlock(256,3, strides=2) # filters, kernel_size, strides (optional, set to 1)
        res7 = resnet_7.call(dropped_out_res6)
        resnet_8 = ResnetBlock(256,3) # filters, kernel_size
        res8 = resnet_8.call(res7)
        resnet_9 = ResnetBlock(256,3) # filters, kernel_size
        res9 = resnet_9.call(res8)
        dropped_out_res9 = self.drop_layer(res9)

        resnet_10 = ResnetBlock(512,3, strides=2) # filters, kernel_size, strides (optional, set to 1)
        res10 = resnet_10.call(dropped_out_res9)
        resnet_11 = ResnetBlock(256,3) # filters, kernel_size
        res11= resnet_11.call(res10)
        resnet_12 = ResnetBlock(256,3) # filters, kernel_size
        res12 = resnet_12.call(res11)
        dropped_out_res12 = self.drop_layer(res12)

        # now do average pooling
        pooling_layer = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
        pooled = pooling_layer(dropped_out_res12)
        dropped_pool = self.drop_layer(pooled)

        dense_layer = tf.keras.layers.Dense(self.num_classes, input_shape=(512,))
        res = dense_layer(dropped_pool)
        print('res shape ', res.shape)


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
        mean_loss = tf.math.reduce_mean(sum) / self.batch_size

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


def get_batch(start_index, batch_size, train_inputs, train_labels):
    input_batch = train_inputs[start_index : (start_index + batch_size)]
    label_batch = train_labels[start_index : (start_index + batch_size)]
    flipped_in = tf.image.random_flip_left_right(input_batch) # mention this in their architecture
    return flipped_in, label_batch


def train(model, train_inputs, train_labels): # from my own CNN project :)
    '''
    Trains the model on all of the inputs and labels for one epoch.
    '''
    #shuffle
    indices = np.arange(model.num_examples)
    indices_shuffled = tf.random.shuffle(indices)
    shuffled_in = tf.gather(train_inputs, indices)
    shuffled_labels = tf.gather(train_labels, indices)


    i = 0
    while i <= (model.num_examples - model.batch_size):
        input_batch, label_batch = get_batch(i, model.batch_size, shuffled_in, shuffled_labels)

        with tf.GradientTape() as tape:
            logits = model.call(input_batch, False)
            loss = model.loss(logits, label_batch)
            accuracy = model.accuracy(logits, label_batch)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        i = i + model.batch_size


    return

# from my own CNN project :)
def test(model, test_inputs, test_labels): # will need to do this for 15 epochs, according to paper
    """
    Tests the model on the test inputs and labels.
    """
    # should we batch these? didn't in cnn but sometimes do, idk why
    logits = model.call(test_inputs, True)
    accuracy = model.accuracy(logits, test_labels)
    print(accuracy)
    pass

def main():
    return


if __name__ == '__main__':
    main()
