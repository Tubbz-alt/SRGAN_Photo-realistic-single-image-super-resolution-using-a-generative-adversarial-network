import tensorflow as tf




class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-9)
        self.conv_act = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.conv_act(x)


class SRDiscriminator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(SRDiscriminator, self).__init__(**kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(filters=64,
                                             kernel_size=3,
                                             strides=1,
                                             padding='same')
        self.conv_1_act = tf.keras.layers.LeakyReLU(alpha=0.2)

        #64, 128, 128, 256, 256, 512, 512
        #2, 1, 2, 1, 2, 1, 2

        self.block_1 = ConvLayer(filters=64, kernel_size=3, strides=2, padding='same')

        self.blocks = []
        for _filter in [128, 256, 512]:
            for _stride in [1, 2]:
                conv_layer = ConvLayer(filters=_filter, kernel_size=3, strides=_stride, padding='same')
                self.blocks.append(conv_layer)


        # QUESTION: SHOULD PUT FLATTEN TO MAP THE IMG OR FOLLOW AS PIX2PIX METHOD?
        self.fc_1 = tf.keras.layers.Dense(1024)
        self.fc_1_act = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.fc_2 = tf.keras.layers.Dense(1)
        self.fc_2_act = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_1_act(x)
        x = self.block_1(x)

        for block in self.blocks:
            x = block(x)

        # if required flatten, add at this position
        x = self.fc_1(x)
        x = self.fc_1_act(x)
        x = self.fc_2(x)
        return self.fc_2_act(x)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    test_discriminator = SRDiscriminator()
    tmp_input = tf.random.normal(shape=(2, 256, 256, 3))

    test_output = test_discriminator(tmp_input)

    test_output.shape

