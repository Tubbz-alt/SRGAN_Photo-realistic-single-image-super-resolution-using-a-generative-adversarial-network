import tensorflow as tf
import os

# Data Shape Checker
def downloaded_img_shape_check(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img


# Shape Check n-th data
def check_data(config, check_n_data):
    shapes = []
    for n, i in enumerate(os.listdir(config['dataset_dir'])):
        data_shape = downloaded_img_shape_check(str(config['dataset_dir'])+'/'+i).shape
        shapes.append(data_shape)
        if n == check_n_data:
            break
    return shapes