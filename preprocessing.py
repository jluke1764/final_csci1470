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

def get_data(data_path, labels_path, flip=False):

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

    # print(len(inputs))

    # reshape image data
    # 128*128

    inputs = np.array(inputs) # (1248, 128, 128)
    labels = np.array(labels)


    # one hot labels
    one_hot_labels = tf.one_hot(labels, num_classes)
    print(one_hot_labels.shape)

    return (inputs, one_hot_labels, my_labels)

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

    train_inputs = inputs_split[0]
    train_labels = labels_split[0]
    test_inputs = inputs_split[1]
    test_labels = labels_split[1]
    print("test input shape", test_inputs.shape)

    print("test labels shape", test_labels.shape)


    new_im = PIL.Image.fromarray(train_inputs[0])
    # print(train_inputs[0])
    new_im.save("test0.png")

    # new_im = PIL.Image.fromarray(train_inputs[1])
    # new_im.save("test1.png")

    # new_im = PIL.Image.fromarray(train_inputs[2])
    # new_im.save("test2.png")

    # new_im = PIL.Image.fromarray(train_inputs[3])
    # new_im.save("test3.png")

    # new_im = PIL.Image.fromarray(train_inputs[4])
    # new_im.save("test4.png")

    train_inputs = np.expand_dims(train_inputs, -1).astype(float) #(1248, 128, 128, 1)
    test_inputs = np.expand_dims(test_inputs, -1).astype(float) #(1248, 128, 128, 1)


    return (train_inputs, train_labels, test_inputs, test_labels)




# (inputs, labels, text_label_list) = get_data("./data", "./my_2_labels.txt")
# split_into_train_test(inputs, labels)



