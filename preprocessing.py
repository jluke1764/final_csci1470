import pickle
import numpy as np
import tensorflow as tf
import os
import PIL
"""
Here we will preprocess the data.
Steps:
-- download the images
-- select particular categories from data to reduce data size (like hw2)
-- match data and labels, return them

potential things to improve performance
-- may need to resize or recenter data (pending....)
-- may need to reduce resolution
"""

def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled" 
	object produced with cPickle. The code below will open up a 
	"pickled" object (each file) and return a dictionary.

	NOTE: DO NOT EDIT

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def get_data(data_path, labels_path):

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
            inputs.append(data)
            labels.append(label_ind)

        label_ind +=1

    print(len(inputs))

    # reshape image data
    # 128*128

    inputs = np.array(inputs) # (1248, 128, 128, 3)
    labels = np.array(labels)

    # print(inputs.shape)
    # print(labels.shape)
    # print(labels[691])
    # print(labels[692])

    new_im = PIL.Image.fromarray(inputs[0])
    new_im.save("cat0.png")

    # print(inputs[0])

    # one hot labels
    one_hot_labels = tf.one_hot(labels, num_classes)
    print(one_hot_labels.shape)

    #normalize inputs
    inputs = np.expand_dims(inputs, -1)
    normalized_inputs = inputs/255
    # print(inputs[0].shape)


    return (normalized_inputs, one_hot_labels)

def split_into_train_test(inputs, labels, frac=.8):

    num_examples = inputs.shape[0]

    train_num = int(np.floor(num_examples*frac))
    print(train_num)

    # shuffle data
    indices = np.arange(num_examples)
    np.random.shuffle(indices)
    inputs = tf.gather(inputs, indices)
    labels = tf.gather(labels, indices)

    inputs_split = np.vsplit(inputs, np.arange(train_num, len(inputs), train_num))
    labels_split = np.vsplit(labels, np.arange(train_num, len(labels), train_num))
    # split_matrix_list = np.vsplit(matrix, np.arange(batch_size, len(matrix), batch_size))


    train_inputs = inputs_split[0]
    train_labels = labels_split[0]
    test_inputs = inputs_split[1]
    test_labels = labels_split[1]

    # print(len(train_inputs))
    print("114", train_labels.shape)
    # print(len(test_inputs))
    # print(len(test_labels))



    return (train_inputs, train_labels, test_inputs, test_labels)




(inputs, labels) = get_data("./data", "./my_2_labels.txt")
split_into_train_test(inputs, labels)



