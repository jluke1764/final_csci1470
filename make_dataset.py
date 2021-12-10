import pickle
import numpy as np
import tensorflow as tf
import os
import PIL


def get_labels(sketches_filepath, data_filepath, make_txt=False):
"""
Returns a list of all labels from the sketch directory. Also creates a txt file if desired.
:param sketches_filepath: the path to the locally downloaded sketch directory
:param data_filepath: the path to where we want to put the generated txt of all the labels
:param make_txt: boolean for whether the txt should be generated
:return: list of all labels
"""

    labels = os.listdir(sketches_filepath)
    if (make_txt == True):
        file_name = "all_labels.txt"
        path = os.path.join(data_filepath, file_name)

        f = open(path, "w+")
        print(labels)
        for l in labels:
            f.write("%s \n" % l)

        f.close()

    return labels

def imgs2bytes(sketches_filepath, label, data_filepath, labels_list):
"""
Pickles the images for a select label into one file
:param sketches_filepath: the path to the locally downloaded sketch directory
:param data_filepath: the path to where we want to put the generated pickle file
:param labels_list: list of all possible labels
"""
    if (label in labels_list == False):
        print("bad label")
        return

    images = os.listdir(sketches_filepath+"/"+label)

    img_bytes_list = []

    f = open(data_filepath+label, "wb")

    data = []
    for image in images:
        img_path = os.path.join(sketches_filepath, label, image) 

        # resize image to 128x128
        img = PIL.Image.open(img_path)
        img = img.resize((128, 128))
        img = img.convert('1') # convert image to black and white

        img_data = np.array(img)
        data.append(img_data)
    
    pickle.dump(data, f)

    f.close()

def make_select_pickles(sketches_filepath, my_labels_filename):
"""
Given a list of labels, generates pickle files from the sketch directory
:param sketches_filepath: the path to the locally downloaded sketch directory
:param my_labels_filename: the name of the txt that has the list of labels we want to create pickle files for
"""

    my_labels_filepath = os.path.join(".", my_labels_filename)


    data_filepath = "./data/"
    if (os.path.exists(data_filepath) == False):
        data_dir = os.mkdir(data_filepath)

    labels_list = get_labels(sketches_filepath, "./")

    with open(my_labels_filepath, "r") as f:
        my_labels = f.readlines()
        my_labels = ' '.join(my_labels).split()
        f.close()

    for label in my_labels:
        print(label)
        label = label.strip()
        imgs2bytes(sketches_filepath, label, data_filepath, labels_list)


def main():
"""
if the pickled data does not already exist locally on your computer, run this with the filepaths where the sketches are kept 
on your machine. Examples commented below
"""
    # sketch_path = "/Users/JackieLuke/cs1470/rendered_256x256/256x256/sketch/tx_000100000000"
    # labels_list = get_labels(sketch_path, "./")
    # make_select_pickles(sketch_path, "all_labels.txt")
    return


if __name__ == '__main__':
    main()




