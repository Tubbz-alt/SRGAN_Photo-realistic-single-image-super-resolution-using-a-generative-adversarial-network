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

# Checkpoint
ckpt = tf.train.Checkpoint(generator=generator,
                           discirminator=discriminator,
                           generator_optimizer=generator_optimizer,
                           discriminator_optimizer=discriminator_optimizer)

checkpoint_manager = tf.train.CheckpointManager(ckpt,
                                                directory = config_dict['log_save_path'],
                                                max_to_keep = config_dict['max_to_keep'],
                                                keep_checkpoint_every_n_hours = config_dict['keep_ckpt_n_hours'])


# Train Step
generator_loss_metrics = tf.keras.metrics.Mean(name='Generator Loss')
discriminator_loss_metrics = tf.keras.metrics.Mean(name='Discriminator Loss')

@tf.function
def train_step(hr_img, lr_img):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        sr_img = generator(lr_img)

        sr_disc = discriminator(sr_img)
        hr_disc = discriminator(hr_img)

        cont_loss = calc_content_loss(hr_img, sr_img)
        gen_loss = calc_gen_loss(sr_img)
        total_loss = cont_loss + (0.001 * gen_loss)

        disc_loss = calc_disc_loss(hr_disc, sr_disc)

    gen_gradient = gen_tape.gradient(total_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))
    generator_loss_metrics.update_state(total_loss)
    discriminator_loss_metrics.update_state(disc_loss)


def train_srgan(EPOCHS):
    if checkpoint_manager.latest_checkpoint:
        ckpt.restore(checkpoint_manager.latest_checkpoint)
        print(f'Restore Latest Checkpoint -- {checkpoint_manager.latest_checkpoint}')
    for e in range(EPOCHS):
        for batch_, (hr_img, lr_img) in enumerate(dataset):
            train_step(hr_img, lr_img)
