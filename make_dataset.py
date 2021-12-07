import pickle
import numpy as np
import tensorflow as tf
import os
import PIL

"""
/Users/JackieLuke/cs1470/rendered_256x256/256x256/sketch/tx_000100000000
sketches_filepath is where the images live in the computer
data_filepath is where I want the generated txt filepath to live
"""
def get_labels(sketches_filepath, data_filepath, make_txt=False):
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

    if (label in labels_list == False):
        print("bad label")
        return

    images = os.listdir(sketches_filepath+"/"+label)

    img_bytes_list = []

    # print(os.getcwd())

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




    # print(images)

# labels_list = get_labels("/Users/JackieLuke/cs1470/rendered_256x256/256x256/sketch/tx_000100000000", "/Users/JackieLuke/cs1470/final_csci1470")

# imgs2bytes("/Users/JackieLuke/cs1470/rendered_256x256/256x256/sketch/tx_000100000000", "cow", "/Users/JackieLuke/cs1470/final_csci1470/data", labels_list)

make_select_pickles("/Users/rachelfuller/Desktop/rendered_256x256/256x256/sketch/tx_000100000000", "my_10_labels.txt")
# make_select_pickles("/Users/JackieLuke/cs1470/rendered_256x256/256x256/sketch/tx_000100000000", "my_2_labels.txt")
