import tensorflow as tf
import tensorflow.keras as keras
from layerlib_stage1 import Conv2D_ReflectPad, Conv2DTranspose, AtrousBlockPad2



class AtrousNet_Test2(keras.Model):
    def __init__(self, num_blocks=10, max_global_stride=8, pad_to_fit_global_stride=True, d_mult=16,
                 activation=tf.nn.relu, atrousDim = [1,2,4,8]):
        super(AtrousNet_Test2, self).__init__()
        self.num_blocks = num_blocks
        self.max_global_stride = max_global_stride
        self.pad_to_fit_global_stride = pad_to_fit_global_stride
        self.d_mult = d_mult
        self.activation = activation
        #self.d_blocks = self.d_mult * self.max_global_stride

        self.downsampling_layers = []

        self.downsampling_layers.append(Conv2D_ReflectPad(filters=self.d_mult,
                                                          kernel_size=7,
                                                          strides=(1, 1),
                                                          use_bias=True,
                                                          use_scale=True,
                                                          padding="same",
                                                          act=self.activation))

        self.downsampling_layers.append(Conv2D_ReflectPad(filters=self.d_mult * 2,
                                                          kernel_size=3,
                                                          strides=(2, 2),
                                                          use_bias=True,
                                                          use_scale=True,
                                                          padding="same",
                                                          act=self.activation))



        self.blocks = []
        for x in range(num_blocks):
            self.blocks.append(AtrousBlockPad2(filters=self.d_mult * 4,
                                               kernel_size=3,
                                               strides=(1, 1),
                                               use_bias=True,
                                               use_scale=True,
                                               activation=self.activation,
                                               atrousBlocks=atrousDim))

        self.upsampling_layers = []
        self.upsampling_layers.append(Conv2DTranspose(filters=self.d_mult * 2,
                                                      kernel_size=3,
                                                      strides=(2, 2),
                                                      padding="same",
                                                      act=self.activation,
                                                      use_bias=True,
                                                      use_scale=True))

        self.output_layer = []
        self.output_layer.append(Conv2D_ReflectPad(filters=self.d_mult,
                                        kernel_size=3,
                                        strides=(1, 1),
                                        use_bias=True,
                                        use_scale=True,
                                        padding="same",
                                        act=self.activation))
        self.output_layer.append(Conv2D_ReflectPad(filters=3,
                                        kernel_size=3,
                                        strides=(1, 1),
                                        use_bias=True,
                                        use_scale=True,
                                        padding="same",
                                        act=None))

    def call(self, input_data):
        downs = []
        net = input_data
        for x in range(len(self.downsampling_layers)):
            net = self.downsampling_layers[x](net)
            downs.append(net)

        for x in range(len(self.blocks)):
            net = self.blocks[x](net)

        for x in range(len(self.upsampling_layers)):
            idx = len(downs) - x - 1
            net = tf.concat([net, downs[idx]], axis=-1)
            net = self.upsampling_layers[x](net)

        for x in range(len(self.output_layer)):
            net = self.output_layer[x](net)

        return input_data + net