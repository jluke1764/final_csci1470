import pickle
import numpy as np
import tensorflow as tf
import os
import PIL


def unpickle(file):
	"""
	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def get_data(data_path, labels_path, flip=False):
    """
    Function to get all data. Not split into test and train in this function.

	:param data_path: filepath to the data folder
	:param labels_path: filepath to the txt file of labels that we want to classify in our model
	:param flip: boolean to add flipped images to the dataset to double the amount of data

	returns
	inputs: inputs of shape (num_examples, 128, 128, 1)
	one_hot_labels: one hot form of labels of shape (num_examples, num_classes)
	my_labels: NOT-one hot form of labels of shape (num_examples, num_classes)

	"""

    # get my labels
    with open(labels_path, "r") as f:
        my_labels = f.readlines()
        my_labels = ' '.join(my_labels).split()
        f.close()

    num_classes = len(my_labels)

    # get data from pickle files by label
    inputs = []
    labels = []
    label_ind = 0
    for my_label in my_labels:
        print(my_label)
        my_label = my_label.strip()
        unpickled_file = unpickle(os.path.join(data_path, my_label))
        for data in unpickled_file:
            # print("datashape", data.shape)
            inputs.append(data)
            labels.append(label_ind)

            if (flip == True):
                inputs.append(np.fliplr(data))
                labels.append(label_ind)

        label_ind +=1

    # reshape image data
    # 128*128
    inputs = np.array(inputs) 
    labels = np.array(labels)

    # one hot labels
    one_hot_labels = tf.one_hot(labels, num_classes)
    return (inputs, one_hot_labels, my_labels)


def split_into_train_test(inputs, labels, frac=.8):
    """
	Splits input data and labels into train and test data based on a given ratio of test/train

	:param inputs: inputs of shape (num_examples, 128, 128, 1)
	:param labels: the one hot labels corresponding to the inputs of shape (num_examples, num_classes)
	:param frac: the ration of train to test you want. For example frac = 0.8 with 100 examples would give you
	80 train and 20 test.

	returns tuple
	    train_inputs of shape (num_examples*frac, 128, 128, 1)
        train_labels of shape (num_examples*frac, num_classes)
	    test_inputs of shape (num_examples*(1-frac), 128, 128, 1)
        test_labels  of shape (num_examples*(1-frac), num_classes)
	"""


    num_examples = inputs.shape[0]

    train_num = int(np.floor(num_examples*frac))

    # shuffle data
    indices = np.arange(num_examples)
    np.random.shuffle(indices)
    inputs = tf.gather(inputs, indices)
    labels = tf.gather(labels, indices)

    inputs_split = np.vsplit(inputs, np.arange(train_num, len(inputs), train_num))
    labels_split = np.vsplit(labels, np.arange(train_num, len(labels), train_num))

    train_inputs = inputs_split[0]
    train_labels = labels_split[0]
    test_inputs = inputs_split[1]
    test_labels = labels_split[1]

    # new_im = PIL.Image.fromarray(train_inputs[0])
    # new_im.save("test0.png")

    train_inputs = np.expand_dims(train_inputs, -1).astype(float) 
    test_inputs = np.expand_dims(test_inputs, -1).astype(float) 

    return (train_inputs, train_labels, test_inputs, test_labels)
