import tensorflow as tf

class LoadVGG:
    def __init__(self, vgg_type: str):
        self.vgg_type = vgg_type

    def load_loss_model(self):
        return VGGModel(self.vgg_type)


class VGGModel(tf.keras.Model):
    def __init__(self, vgg_type, **kwargs):
        super(VGGModel, self).__init__(**kwargs)
        self.vgg_type = vgg_type
        self.vgg_full_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    def _preprocess_input(self, x):
        x = tf.cast(x, tf.float32)
        norm_result = x / 127.5 - 1
        return norm_result

    def _slice_vgg(self):
        if self.vgg_type == '22':
            vgg_loss_model = self.vgg_full_model.layers[:5]
        else:
            vgg_loss_model = self.vgg_full_model.layers[:-1]

        vgg_loss_model = tf.keras.Sequential(vgg_loss_model)
        vgg_loss_model.trainable = False
        return vgg_loss_model

    def call(self, inputs):
        prep = tf.keras.layers.Lambda(self._preprocess_input)(inputs)
        x = self._slice_vgg()(prep)
        return x



if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    test_input = tf.random.uniform(shape=(2, 256, 256, 3))

    tmp_model_22 = LoadVGG('22').load_loss_model()
    tmp_model_22_out = tmp_model_22(test_input)
    tmp_model_22_out.shape # 2, 128, 128, 128

    tmp_model_54 = LoadVGG('54').load_loss_model()
    tmp_model_54_out = tmp_model_54(test_input)
    tmp_model_54_out.shape # 2, 16, 16, 512






