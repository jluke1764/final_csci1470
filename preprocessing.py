<<<<<<< HEAD
import pickle
import numpy as np
import tensorflow as tf
import os
import PIL
from PIL import image


"""
resize images to specifized shape, turn image files into bytes
"""
def make_datasets(filepath, width, height):



=======
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

import gzip
import numpy as np
import gzip

def get_data(inputs_file_path):
    # Step 1: 
>>>>>>> 3e498e8c2039b5ba8a1956516c6e8ec19ceb3296
