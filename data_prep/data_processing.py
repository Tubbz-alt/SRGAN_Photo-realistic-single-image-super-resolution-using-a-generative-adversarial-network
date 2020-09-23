import ruamel.yaml
import os
import tensorflow as tf

from utils.load_config import LoadConfig
from utils.img_shape_checker import check_data

import matplotlib.pyplot as plt

# To use the repo, requires to
assert tf.__version__ >= '2.3.0', 'This Repository requires Tensorflow >= 2.3.0'


# Data Prepare Strategy
# 1) Create High-resolution data
# 2) Create Low-resolution data
#  Data augmentation occur in architecture
# 3) Add data augmentation, use flip, rotation
# 4) Data Scaling performes in the architecture, because it requires to use VGG architecture for loss calculation

# V-1
# DataGenerator Pipeline from tf.keras.image
class DataGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.num_of_imgs = len(os.listdir(self.config['dataset_dir']))

    def load_img(self, img_path):
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        return img

    def image_generator(self):
        for img_path in os.listdir(self.config['dataset_dir']):
            yield self.load_img(os.path.join(self.config['dataset_dir'], img_path))


class DataProcessor:
    def __init__(self, config: dict, generator):
        self.config = config
        self.generator = generator

    def create_hr_img(self, img):
        img = tf.cast(img, tf.float32)
        hr_img = tf.image.resize(img,
                                 size=(self.config['H_img_height'], self.config['H_img_width']),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return hr_img

    def create_lr_img(self, img):
        img = tf.cast(img, tf.float32)
        lr_img = tf.image.resize(img,
                                 size=(self.config['L_img_height'], self.config['L_img_width']),
                                 method=tf.image.ResizeMethod.BICUBIC)
        return lr_img


class DataCreator:
    def __init__(self, config: dict):
        self.config = config
        self.generator = DataGenerator(self.config)
        self.dataprocessor = DataProcessor(self.config, self.generator)
        self.data_shape = check_data(self.config, 1)

    def _check_img_result(self, dataset):
        for n, (hr_data, lr_data) in enumerate(dataset.take(1)):
            plt.subplot(1,2,1)
            plt.imshow(hr_data.numpy().astype('int'))
            plt.title('HR_DATA: (256, 256)')
            plt.subplot(1,2,2)
            plt.imshow(lr_data.numpy().astype('int'))
            plt.title('LR_DATA: (64, 64)')
            plt.axis('off')
        plt.show()

    def create_data(self,
                    batch_size: int,
                    shuffle: bool,
                    check_result: bool,
                    save_tf: bool):
        try:


                full_dataset = tf.data.experimental.load(self.config['dataset_save_path'],
                                                         element_spec=(tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
                                                                       tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32)))
                return full_dataset
        except:
            dataset = tf.data.Dataset.from_generator(self.generator.image_generator,
                                                     output_types=tf.float32,
                                                     output_shapes=(self.data_shape[0][0],
                                                                    self.data_shape[0][1],
                                                                    self.data_shape[0][2]))

            hr_dataset = dataset.map(self.dataprocessor.create_hr_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            lr_dataset = dataset.map(self.dataprocessor.create_lr_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            full_dataset = tf.data.Dataset.zip((hr_dataset, lr_dataset))

            if check_result:
                self._check_img_result(full_dataset)

            if shuffle:
                full_dataset = full_dataset.shuffle(buffer_size=self.generator.num_of_imgs)

            full_dataset = full_dataset.batch(batch_size)
            full_dataset = full_dataset.prefetch(tf.data.experimental.AUTOTUNE)

            if save_tf:
                tf.data.experimental.save(full_dataset,
                                          self.config['dataset_save_path'],
                                          compression=None)
            return full_dataset


if __name__ == '__main__':

    config_path = './config'
    config = LoadConfig(config_path)()

    config

    dataset = DataCreator(config)

    dataset.create_data(10, True, True, False)
