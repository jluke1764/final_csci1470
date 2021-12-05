"""
This is where we will train, given whichever model is requested. We will call functions
from the preprocessing step and the actual model here.
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
from model import Model
from preprocessing import get_data
from preprocessing import split_into_train_test

def get_batch(start_index, batch_size, train_inputs, train_labels):
    input_batch = train_inputs[start_index : (start_index + batch_size)]
    label_batch = train_labels[start_index : (start_index + batch_size)]
    flipped_in = tf.image.random_flip_left_right(input_batch) # mention this in their architecture
    return flipped_in, label_batch


def train(model, train_inputs, train_labels, num_examples): # from my own CNN project :)
    '''
    Trains the model on all of the inputs and labels for one epoch.
    '''
    #shuffle
    # print('TRAIN LABELS SIZE', train_labels.shape)
    indices = tf.convert_to_tensor(np.arange(num_examples))
    # print(indices.shape)
    indices_shuffled = tf.random.shuffle(indices)
    shuffled_in = tf.gather(train_inputs, indices)
    shuffled_labels = tf.gather(train_labels, indices)

    


    i = 0
    while i <= (num_examples - model.batch_size):
        print("batch number ", i/64)
        input_batch, label_batch = get_batch(i, model.batch_size, shuffled_in, shuffled_labels)
        input_batch = tf.convert_to_tensor(input_batch)
        label_batch = tf.convert_to_tensor(label_batch)

        with tf.GradientTape() as tape:
            logits = model.call(input_batch)
            loss = model.loss(logits, label_batch)
            accuracy = model.accuracy(logits, label_batch)
            print('loss ', loss, ' accuracy ', accuracy)


        #print('TRAINABLE VARIABLES ', len(model.trainable_variables))
        #print('TRAINABLE VARIABLES ', [v.name for v in model.trainable_variables])
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
    print('Click clack moo')

    (inputs, labels) = get_data("./data", "./my_2_labels.txt")
    (train_inputs, train_labels, test_inputs, test_labels) = split_into_train_test(inputs, labels)
    print("TRAIN LABELS SHAPE: ", train_labels.shape)

    my_model = Model()
    # test_tensor = tf.zeros((10, 128, 128, 1))
    # logits = my_model.call(test_tensor)
    for i in range(15):
        print("epoch ", i)
        train(my_model, train_inputs, train_labels, 998)

    return

if __name__ == '__main__':
    main()
