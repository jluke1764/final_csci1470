"""
This is where we will train, given whichever model is requested. We will call functions
from the preprocessing step and the actual model here.
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
from model import ResnetModel
from basic_model import BasicModel
from preprocessing import get_data
from preprocessing import split_into_train_test
import sys
import PIL
import time

def get_batch(start_index, batch_size, train_inputs, train_labels):
    input_batch = train_inputs[start_index : (start_index + batch_size)]
    label_batch = train_labels[start_index : (start_index + batch_size)]
    flipped_in = tf.image.random_flip_left_right(input_batch) # include randomly flipping
    return flipped_in, label_batch


def train(model, train_inputs, train_labels, num_examples): # from class CNN project
    '''
    Trains the model on all of the inputs and labels for one epoch.
    '''
    indices = tf.convert_to_tensor(np.arange(num_examples)) # make sure is tensor so gradients flow
    indices_shuffled = tf.random.shuffle(indices)
    shuffled_in = tf.gather(train_inputs, indices)
    shuffled_labels = tf.gather(train_labels, indices)

    loss_list = []
    i = 0
    while i <= (num_examples - model.batch_size):
        input_batch, label_batch = get_batch(i, model.batch_size, shuffled_in, shuffled_labels)
        input_batch = tf.convert_to_tensor(input_batch)
        label_batch = tf.convert_to_tensor(label_batch)

        with tf.GradientTape() as tape:
            logits = model.call(input_batch)
            loss = model.loss(logits, label_batch)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        accuracy = model.accuracy(logits, label_batch)
        print('batch: ', int(i / model.batch_size), ' loss: ', loss, ' accuracy: ', accuracy)

        i = i + model.batch_size
        loss_list.append(loss)

    return loss_list

# from class CNN project
def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.
    """
    logits = model.call(test_inputs)
    accuracy = model.accuracy(logits, test_labels)
    print("TOP 3 ACC:", top_n_accuracy(logits, test_labels, 3))
    return accuracy


def top_n_accuracy(logits, labels, top_n):
    """
    Computes accuracy. Instead of counting a label as correct only if that label is the #1 most probable,
    this counts the label as correct if it is within the top_n most probably labels. For example,
    an image that is a bear with top three labels [hot-air_balloon, hermit_crab, bear] would be correct if top_n=3
    and incorrect if top_n = 1.

    logits: a matrix of shape (num_inputs, num_classes)
    labels: labels of shape (num_inputs, num_classes)
    top_n: If correct label is among top_n most probable labels, count as correct
    """
    if (logits.shape[1] < top_n):
        top_n = logits.shape[1]

    total_correct = 0

    for l in range(logits.shape[0]):
        img_logits = np.array(logits[l])
        img_label = labels[l]
        preds = []
        for i in range(top_n):
            preds.append(int(tf.argmax(img_logits)))
            img_logits[preds[i]] = -np.inf

        if tf.argmax(img_label) in preds:
            total_correct += 1

    return total_correct/logits.shape[0]


def show_predictions(model, image, logits, label, text_label_list, filename):
    """
    Shows more information on prediction for a single image. Prints out the top three most likely
    categories predicted by the model. Also saves the image as a file.

    image: single image to get info about
    logits: a matrix of shape (num_inputs, num_classes)
    labels: labels of shape (num_inputs, num_classes)
    text_label_list: labels in the form of their names (NOT one hot form)
    filename: filename where the image will be saved

    """
    image = np.squeeze(image, -1).astype(bool)

    img = PIL.Image.fromarray(image)

    # print(logits) can print the logits here, if you want to look at them

    logits = np.array(logits)

    num_predictions = 3
    if (logits.size < num_predictions):
        num_predictions = logits.size
    preds = []
    caption = text_label_list[int(tf.argmax(label))] + ": "

    for i in range(num_predictions):
        preds.append(int(tf.argmax(logits)))
        logits[preds[i]] = -np.inf
        caption += text_label_list[preds[i]] + " "

    img.save(filename)
    print(caption)

def visualize_imgs(model, inputs, labels, text_label_list, num_images):
    """
    Runs the given inputs through the call function and then calls show_predictions
    on each. Does not actually show images, but does save them as files.

    inputs: images to get prediction info about (of shape num_inputs, 128, 128, 1)
    labels: labels of shape (num_inputs, num_classes)
    text_label_list: labels in the form of their names (NOT one hot form)
    num_images: int number of images you want prediction information about (example, the first 10, the first 5, etc.)

    """
    logits = model.call(inputs)
    for p in range(num_images):
        show_predictions(model, inputs[p], logits[p], labels[p], text_label_list, str(p)+".png")


def main(num_epochs):
    start = time.time()

    if len(sys.argv) != 3 or sys.argv[1] not in {"BASIC", "RESNET"}:
        print("USAGE: python assignment.py <Model Type: [BASIC/RESNET]> <labels_file>")
        exit()

    # change labels_file here to run on different labels
    # labels_file should be a text file containing the names of the labels that you want to run the model on,
    # with one category per line
    my_labels_txt = sys.argv[2]


    (inputs, labels, text_label_list) = get_data("./data", my_labels_txt, flip=False)
    (train_inputs, train_labels, test_inputs, test_labels) = split_into_train_test(inputs, labels)

    img = np.squeeze(train_inputs[0], -1).astype(bool)
    new_im = PIL.Image.fromarray(img)
    new_im.save("test01.png")

    num_classes = train_labels.shape[1]
    num_images = train_labels.shape[0]

    if (sys.argv[1] == "BASIC"):
        my_model = BasicModel(num_classes)
    elif (sys.argv[1] == "RESNET"):
        my_model = ResnetModel(num_classes)


    # prints the test accuracy before the model is trained
    print("INITIAL TEST ACCURACY: ", test(my_model, test_inputs, test_labels))

    # allows you to print out the architecture of the basic model, if you want to
    #if (sys.argv[1] == "BASIC"):
        #my_model.convLayers.summary()
        #my_model.denseLayers.summary()

    # train
    loss_list_all = []
    for i in range(num_epochs):
        print("EPOCH ", i)
        loss_list = train(my_model, train_inputs, train_labels, num_images)
        loss_list_all.append(loss_list)

    #test
    print("TEST ACCURACY: ", test(my_model, test_inputs, test_labels))
    print("NUM CLASSES: ", num_classes, " random accuracy is ", 1/num_classes)

    end = time.time()
    print("ELASPED TIME:", end - start)

    visualize_imgs(my_model, test_inputs, test_labels, text_label_list, 10) # tells you top three categories and saves images in files
    return

if __name__ == '__main__':

    num_epochs = 15
    main(num_epochs)
