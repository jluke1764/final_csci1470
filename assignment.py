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

    loss_list = []
    i = 0
    while i <= (num_examples - model.batch_size):
        print("batch number ", i/64)
        input_batch, label_batch = get_batch(i, model.batch_size, shuffled_in, shuffled_labels)
        input_batch = tf.convert_to_tensor(input_batch)
        label_batch = tf.convert_to_tensor(label_batch)

        with tf.GradientTape() as tape:
            logits = model.call(input_batch)
            loss = model.loss(logits, label_batch)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        accuracy = model.accuracy(logits, label_batch)
        print('loss ', loss, ' accuracy ', accuracy)

        i = i + model.batch_size

        
        loss_list.append(loss)

    return loss_list

# from my own CNN project :)
def test(model, test_inputs, test_labels): # will need to do this for 15 epochs, according to paper
    """
    Tests the model on the test inputs and labels.
    """
    # should we batch these? didn't in cnn but sometimes do, idk why

    # print("test input shape", test_inputs.shape)
    # print("test labels shape", test_labels.shape)

    logits = model.call(test_inputs)
    accuracy = model.accuracy(logits, test_labels)
    print("TOP 3 ACC:", top_n_accuracy(logits, test_labels, 3))

    return accuracy

def top_n_accuracy(logits, labels, top_n): # will need to do this for 15 epochs, according to paper
    """
    Tests the model on the test inputs and labels.
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




def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  

def show_predictions(model, image, logits, label, text_label_list, filename):
    image = np.squeeze(image, -1).astype(bool)

    img = PIL.Image.fromarray(image)
    # draw = PIL.ImageDraw.draw(img)

    print(logits)

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



    # first_pred = int(tf.argmax(logits))
    # # print(first_pred)
    # logits[first_pred] = -np.inf
    # second_pred = int(tf.argmax(logits))
    # logits[second_pred] = -np.inf
    # third_pred = int(tf.argmax(logits))

    # draw.text((new_width / 15 + 25, new_height - 100),
    #                        caption, (255, 0, 0), 
    #                        align ="center")

    img.save(filename)
    print(caption)

def visualize_imgs(model, inputs, labels, text_label_list, num_images):
    logits = model.call(inputs)
    for p in range(num_images):
        # show_predictions(model, image, logits, label, text_label_list)
        show_predictions(model, inputs[p], logits[p], labels[p], text_label_list, str(p)+".png")



def main(my_labels_txt, num_epochs):
    print('Click clack moo')

    start = time.time()



    if len(sys.argv) != 2 or sys.argv[1] not in {"BASIC", "RESNET"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [BASIC/RESNET]")
        exit()

    (inputs, labels, text_label_list) = get_data("./data", my_labels_txt, flip=False)
    (train_inputs, train_labels, test_inputs, test_labels) = split_into_train_test(inputs, labels)
    # print("TRAIN LABELS SHAPE: ", train_labels.shape)

    print("MAKING AN IMAGE")
    # print(train_inputs[0].shape)
    img = np.squeeze(train_inputs[0], -1).astype(bool)
    new_im = PIL.Image.fromarray(img)
    # new_im.convert('RGB')
    new_im.save("test01.png")



    num_classes = train_labels.shape[1]


    if (sys.argv[1] == "BASIC"):
        my_model = BasicModel(num_classes)
    elif (sys.argv[1] == "RESNET"):
        my_model = ResnetModel(num_classes)

    # test_tensor = tf.zeros((10, 128, 128, 1))
    # logits = my_model.call(test_tensor)

    # visualize_imgs(my_model, test_inputs, test_labels, text_label_list, 10)
    print("INITIAL TEST ACCURACY: ", test(my_model, test_inputs, test_labels))

    if (sys.argv[1] == "BASIC"):
        my_model.convLayers.summary()
        my_model.denseLayers.summary()

    # train
    loss_list_all = []
    for i in range(num_epochs):
        print("epoch ", i)
        loss_list = train(my_model, train_inputs, train_labels, 998)
        loss_list_all.append(loss_list)

    # loss_list_all = np.flatten(loss_list_all)
    #test
    print("TEST ACCURACY: ", test(my_model, test_inputs, test_labels))

    print("NUM CLASSES: ", num_classes, ";random accuracy is ", 1/num_classes)

    end = time.time()
    print("ELASPED TIME:", end - start)  

    visualize_imgs(my_model, test_inputs, test_labels, text_label_list, 10)
    return

if __name__ == '__main__':

    num_epochs = 15
    filepath = "./my_10_labels.txt"

    main(filepath, num_epochs)
