import tensorflow as tf


# Generator Loss - Content Loss
# L_MSE
def pixel_wise_mse(y_true, y_pred, reduction_factor=4):
    mse_loss = tf.keras.losses.MeanSquaredError()
    result = mse_loss(y_true, y_pred)
    return 1/(reduction_factor **2) * result


def vgg_loss(vgg_model, y_true, y_pred, rescale_factor=1/12.75):
    mse_loss = tf.keras.losses.MeanSquaredError()
    vgg_y_true = vgg_model(y_true)
    vgg_y_pred = vgg_model(y_pred)
    result = mse_loss(vgg_y_true, vgg_y_pred)
    return rescale_factor * result
