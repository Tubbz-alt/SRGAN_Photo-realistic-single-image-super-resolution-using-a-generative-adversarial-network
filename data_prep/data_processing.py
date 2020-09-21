import ruamel.yaml
import os
import tensorflow as tf

from utils.load_config import LoadConfig
from utils.img_shape_checker import check_data

import matplotlib.pyplot as plt

# Sample Restore to check
config = LoadConfig('./config')()
sample_img = r'E:\Data_Hub\CelebA_dataset\img_align_celeba\000001.jpg'


# Data Prepare Strategy
# 1) Create High-resolution data
# 2) Create Low-resolution data
#  Data augmentation occur in architecture
# 3) Add data augmentation, use flip, rotation
# 4) Data Scaling performes in the architecture, because it requires to use VGG architecture for loss calculation

# TODO: REQUIRES TO WRAP WITH CLASS

def load_img(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img

def img_generator():
    config = LoadConfig('./config')()
    img_dir = config['dataset_dir']
    for img_path in os.listdir(img_dir):
        yield load_img(os.path.join(img_dir, img_path))


def create_hr_img(config, img):
    img = tf.cast(img, tf.float32)
    hr_img = tf.image.resize(img, size=(config['H_img_height'], config['H_img_width']), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return hr_img


def create_lr_img(config, img):
    img = tf.cast(img, tf.float32)
    lr_img = tf.image.resize(img, size=(config['L_img_height'], config['L_img_width']), method=tf.image.ResizeMethod.BICUBIC)
    return lr_img




config = LoadConfig('./config')()
downloaded_data = check_data(config, 1)

dataset = tf.data.Dataset.from_generator(img_generator,
                                         output_types=tf.int64,
                                         output_shapes=(downloaded_data[0][0],
                                                        downloaded_data[0][1],
                                                        downloaded_data[0][2]))

hr_dataset = dataset.map(lambda x : create_hr_img(config, x))
lr_dataset = dataset.map(lambda x: create_lr_img(config, x))
zip_data = tf.data.Dataset.zip((hr_dataset, lr_dataset))



# Data Checker
def show_pair_img(dataset):
    for n, (hr, lr) in enumerate(dataset.take(1)):
        plt.subplot(1, 2, 1)
        plt.imshow(hr.numpy().astype('int'))
        plt.title('High Resolution : (256, 256)')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(lr.numpy().astype('int'))
        plt.title('Low Resolution : (64, 64)')
        plt.axis('off')
    plt.show()