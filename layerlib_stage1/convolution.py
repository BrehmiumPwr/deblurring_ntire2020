import tensorflow as tf
import numpy as np
from .padding import Padding


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init

    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        weight = tf.random.normal(shape=shape, mean=0.0, stddev=1)
        var = tf.Variable(initial_value=weight, dtype=tf.float32) * wscale
    else:
        weight = tf.random.normal(shape=shape, mean=0.0, stddev=std)
        var = tf.Variable(initial_value=weight, dtype=tf.float32)
    return var


class ConvAftermath(tf.keras.layers.Layer):
    def __init__(self, use_bias=True, use_scale=True, norm=None, act=None):
        self.filters = -1
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.norm = norm
        self.act = act
        self.b = None
        self.s = None
        super(ConvAftermath, self).__init__()

    def build(self, input_shape):
        self.filters = input_shape[-1]
        if self.use_bias:
            self.b = tf.Variable(initial_value=tf.zeros(shape=(self.filters)), dtype=tf.float32)
        if self.use_scale:
            self.s = tf.Variable(initial_value=tf.ones(shape=(self.filters)), dtype=tf.float32)
        super(ConvAftermath, self).build(input_shape)

    def call(self, inputs):
        net = inputs
        if self.use_bias:
            net = net + self.b
        if self.use_scale:
            net = net * self.s
        if self.norm is not None:
            net = self.norm(net)
        if self.act is not None:
            net = self.act(net)
        return net


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), use_bias=True, use_scale=True, norm=None, act=None,
                 padding="same"):
        self.padding = padding
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.strides = strides
        self.kernel_size = kernel_size
        self.filters = filters
        self.atrous_rate = 1

        self.conv = tf.keras.layers.Conv2D(filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           use_bias=False,
                                           activation=None,
                                           padding=self.padding)
        self.conv_aftermath = ConvAftermath(use_bias=self.use_bias, use_scale=self.use_scale,
                                            norm=self.norm, act=self.act)
        super(Conv2D, self).__init__()

    def call(self, inputs):
        net = self.conv(inputs)
        return self.conv_aftermath(net)


class Conv2D_ReflectPad(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), use_bias=True, use_scale=True, norm=None, act=None,
                 padding="same", padding_algorithm="REFLECT"):
        self.padding = padding
        self.padding_algorithm = padding_algorithm
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.strides = strides
        self.kernel_size = kernel_size
        self.filters = filters
        self.atrous_rate = 1

        self.virtual_kernel_size = (self.atrous_rate * (self.kernel_size - 1)) + 1
        self.pad = Padding(ks=self.virtual_kernel_size, s=self.strides[0], algorithm=self.padding_algorithm)

        self.conv = tf.keras.layers.Conv2D(filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           use_bias=False,
                                           activation=None,
                                           padding="VALID") #padding=self.padding)
        self.conv_aftermath = ConvAftermath(use_bias=self.use_bias, use_scale=self.use_scale,
                                            norm=self.norm, act=self.act)
        super(Conv2D_ReflectPad, self).__init__()

    def call(self, inputs):
        net = self.pad(inputs)
        net = self.conv(net)
        return self.conv_aftermath(net)


# Atous Convolution, similar to Conv2D
class AtrousConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), use_bias=True, use_scale=True, norm=None, act=None,
                 padding="same", dilation=1):
        self.padding = padding
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.strides = strides
        self.kernel_size = kernel_size
        self.filters = filters

        self.conv = tf.keras.layers.Conv2D(filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           use_bias=False,
                                           activation=None,
                                           dilation_rate=dilation,
                                           padding=self.padding)
        self.conv_aftermath = ConvAftermath(use_bias=self.use_bias, use_scale=self.use_scale,
                                            norm=self.norm, act=self.act)
        super(AtrousConv2D, self).__init__()

    def call(self, inputs):
        net = self.conv(inputs)
        return self.conv_aftermath(net)

# Atous Convolution, similar to Conv2D
class AtrousConv2D_ReflectPad(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), use_bias=True, use_scale=True, norm=None, act=None,
                 padding_algorithm="REFLECT", dilation=1):
        self.padding_algorithm = padding_algorithm
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.strides = strides
        self.kernel_size = kernel_size
        self.filters = filters
        self.dilation = dilation

        self.virtual_kernel_size = (self.dilation * (self.kernel_size - 1)) + 1
        self.pad = Padding(ks=self.virtual_kernel_size, s=self.strides[0], algorithm=self.padding_algorithm)

        self.conv = tf.keras.layers.Conv2D(filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           use_bias=False,
                                           activation=None,
                                           dilation_rate=self.dilation,
                                           padding="VALID")
        self.conv_aftermath = ConvAftermath(use_bias=self.use_bias, use_scale=self.use_scale,
                                            norm=self.norm, act=self.act)
        super(AtrousConv2D_ReflectPad, self).__init__()

    def call(self, inputs):
        net = self.pad(inputs)
        net = self.conv(net)
        return self.conv_aftermath(net)

class Conv2DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), atrous_rate=1, use_bias=True, use_scale=True, norm=None,
                 w_norm=None, act=None, padding="same"):
        self.padding = padding.upper()
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.strides = strides
        self.atrous_rate = [1, atrous_rate, atrous_rate, 1]
        self.kernel_size = kernel_size
        self.filters = filters
        self.weight = 0

        self.conv_aftermath = ConvAftermath(use_bias=self.use_bias, use_scale=self.use_scale,
                                            norm=self.norm, act=self.act)
        if w_norm is not None:
            self.w_norm = w_norm()
        else:
            self.w_norm = lambda x, training=True: x
        super(Conv2DTranspose, self).__init__()

    def build(self, input_shape):
        kernel_shape = (self.kernel_size, self.kernel_size, self.filters, input_shape[-1])
        self.weight = get_weight(shape=kernel_shape)
        super(Conv2DTranspose, self).build(input_shape)

    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        output_shape = [input_shape[0], input_shape[1] * self.strides[0], input_shape[2] * self.strides[1],
                        self.filters]
        w = self.weight
        if self.strides[0] == 2:
            w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
            w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
        net = tf.nn.conv2d_transpose(inputs, self.w_norm(w, training=training), output_shape=output_shape,
                                     strides=[1, self.strides[0], self.strides[1], 1],
                                     dilations=self.atrous_rate, padding=self.padding,
                                     data_format='NHWC')
        return self.conv_aftermath(net, training=training)



class DepthwiseConv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides=(1, 1), use_bias=True, use_scale=True, norm=None, act=None,
                 padding="same"):
        self.padding = padding
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.strides = strides
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.DepthwiseConv2D(kernel_size=self.kernel_size,
                                                    strides=self.strides,
                                                    use_bias=False,
                                                    activation=None,
                                                    padding=self.padding)
        self.conv_aftermath = ConvAftermath(use_bias=self.use_bias, use_scale=self.use_scale,
                                            norm=self.norm, act=self.act)
        super(DepthwiseConv2D, self).__init__()

    def call(self, inputs):
        net = self.conv(inputs)
        return self.conv_aftermath(net)


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.dense1 = tf.keras.layers.Dense(units=self.filters / self.ratio,
                                            activation=tf.nn.relu)

        self.dense2 = tf.keras.layers.Dense(units=self.filters,
                                            activation=tf.nn.sigmoid)
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        pools = avg_pool + max_pool
        net = self.dense1(pools)
        scale = self.dense2(net)
        return inputs * scale


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        self.conv = tf.keras.layers.Conv2D(filters=1,
                                           kernel_size=self.kernel_size,
                                           strides=[1,1],
                                           use_bias=False,
                                           activation=None,
                                           padding="SAME")

    def build(self, input_shape):
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[3], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[3], keepdims=True)
        concat = tf.keras.layers.concatenate([avg_pool, max_pool])
        scale = self.conv(concat)
        scale = tf.sigmoid(scale)
        return inputs * scale