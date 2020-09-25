import tensorflow as tf
from utils.load_config import LoadConfig
from data_prep.data_processing import DataCreator
from architecture.generator import SRGenerator
from architecture.discriminator import SRDiscriminator
from architecture.load_vgg import VGGModel

from trainer.losses import pixel_wise_mse, vgg_loss

# Load Config
config_dict = LoadConfig('./config')()

# Load Dataset
dataset = DataCreator(config_dict).create_data(batch_size=config_dict['batch_size'],
                                               shuffle=False,
                                               check_result=False,
                                               augmentation=False,
                                               save_tf=False)

# Call architectures
generator = SRGenerator(n_res_layers=16)
discriminator = SRDiscriminator()

# Call vgg model to calculate content loss
vgg_model = VGGModel(config_dict['vgg_loss_model'])


# Define Loss Object
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1)

# Loss Calculate helper
def calc_content_loss(hr_img, sr_img):
    content_loss = vgg_loss(vgg_model, hr_img, sr_img)
    return content_loss


def calc_gen_loss(sr_img):
    loss_result = loss_obj(tf.ones_like(sr_img), sr_img)
    return loss_result


def calc_disc_loss(disc_hr_img, disc_sr_img):
    real_out = loss_obj(tf.ones_like(disc_hr_img), disc_hr_img)
    fake_out = loss_obj(tf.zeros_like(disc_sr_img), disc_sr_img)
    total_disc_loss = real_out + fake_out
    return total_disc_loss


# Call Optimizer
generator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9)

