import tensorflow as tf
import numpy as np
from .padding import Padding
from .dropout import Dropout

def kernel_rotation(kernel):
    # kernel is ksxksxinxout
    rotate90 = tf.transpose(kernel, [1, 0, 2, 3])[:, ::-1, :, :]
    rotate180 = kernel[::-1, ::-1, :, :]
    rotate270 = tf.transpose(kernel, [1, 0, 2, 3])[::-1, :, :, :]
    return [kernel, rotate90, rotate180, rotate270]


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
    def __init__(self, use_bias=True, use_scale=True, norm=None, act=None, dropout_rate=0.0):
        self.filters = -1
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate=self.dropout_rate)
        if norm is not None:
            self.norm = norm()
        else:
            self.norm = None
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

    def call(self, inputs, training=None):
        net = inputs
        if self.use_bias:
            net = net + self.b
        if self.use_scale:
            net = net * self.s
        if self.norm is not None:
            net = self.norm(net, training=training)
        if self.act is not None:
            net = self.act(net)
        if self.dropout_rate > 0.0:
            net = self.dropout(net, training=training)
        return net


class OffsetConvolution(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), atrous_rate=1, use_bias=True, use_scale=True, norm=None,
                 w_norm=None, act=None, padding="same", rotate_filters=False, padding_algorithm="REFLECT"):
        self.padding = padding.upper()
        self.padding_algorithm = padding_algorithm.upper()
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.strides = strides
        self.atrous_rate = [1, atrous_rate, atrous_rate, 1]
        self.kernel_size = kernel_size
        self.filters = filters
        self.rotate_filters = rotate_filters
        self.weight = 0

        self.virtual_kernel_size = (atrous_rate * (self.kernel_size - 1)) + 1

        if self.padding == "SAME":
            self.pad = Padding(ks=self.virtual_kernel_size, s=self.strides[0], algorithm=self.padding_algorithm)
        else:
            self.pad = lambda x: x
        self.conv_aftermath = ConvAftermath(use_bias=self.use_bias, use_scale=self.use_scale,
                                            norm=self.norm, act=self.act)
        if w_norm is not None:
            self.w_norm = w_norm()
        else:
            self.w_norm = lambda x, training: x
        super(OffsetConvolution, self).__init__()

    def offset_output(self, features):
        right_features = tf.pad(features, paddings=[[0, 0], [0, 0], [2, 0],  [0, 0]])[:,:,:-2,:]
        left_features = tf.pad(features, paddings=[[0, 0], [0, 0], [0, 2],  [0, 0]])[:,:,2:,:]
        top_features = tf.pad(features, paddings=[[0, 0], [2, 0], [0, 0], [0, 0]])[:,:-2,:,:]
        bottom_features = tf.pad(features, paddings=[[0, 0], [0, 2], [0, 0], [0, 0]])[:,2:,:,:]
        return tf.concat([features, right_features, left_features, top_features, bottom_features], axis=-1)

    def build(self, input_shape):
        kernel_shape = (self.kernel_size, self.kernel_size, input_shape[-1], self.filters)
        self.weight = get_weight(shape=kernel_shape)

        super(OffsetConvolution, self).build(input_shape)

    def call(self, inputs, training=None):
        net = self.pad(inputs)
        w = self.weight
        if self.rotate_filters:
            w = kernel_rotation(w)
        if self.strides[0] == 2:
            w = tf.pad(self.weight, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
            w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
        net = tf.nn.conv2d(net, self.w_norm(w, training=training), strides=[1, self.strides[0], self.strides[1], 1],
                           dilations=self.atrous_rate, padding="VALID",
                           data_format='NHWC')
        net = self.offset_output(net)
        return self.conv_aftermath(net, training=training)





class Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), atrous_rate=1, use_bias=True, use_scale=True, norm=None,
                 w_norm=None, act=None, padding="same", rotate_filters=False, dropout=0.0, padding_algorithm="REFLECT"):
        self.padding = padding.upper()
        self.padding_algorithm = padding_algorithm.upper()
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.strides = strides
        self.atrous_rate = [1, atrous_rate, atrous_rate, 1]
        self.kernel_size = kernel_size
        self.filters = filters
        self.rotate_filters = rotate_filters
        self.weight = 0
        self.dropout = dropout

        self.virtual_kernel_size = (atrous_rate * (self.kernel_size - 1)) + 1

        if self.padding == "SAME":
            self.pad = Padding(ks=self.virtual_kernel_size, s=self.strides[0], algorithm=self.padding_algorithm)
        else:
            self.pad = lambda x: x
        self.conv_aftermath = ConvAftermath(use_bias=self.use_bias, use_scale=self.use_scale,
                                            norm=self.norm, act=self.act, dropout_rate=self.dropout)

        if w_norm is not None:
            self.w_norm = w_norm()
        else:
            self.w_norm = lambda x, training: x
        super(Conv2D, self).__init__()

    def build(self, input_shape):
        input_channels = input_shape[-1]
        kernel_shape = (self.kernel_size, self.kernel_size, input_channels, self.filters)
        self.weight = get_weight(shape=kernel_shape)
        super(Conv2D, self).build(input_shape)

    def call(self, inputs, training=None):
        net = self.pad(inputs)
        w = self.weight
        if self.rotate_filters:
            rotated_kernels = kernel_rotation(w)
            outputs = []
            for x in range(len(rotated_kernels)):
                w = rotated_kernels[x]
                if self.strides[0] == 2:
                    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
                    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
                outputs.append(tf.nn.conv2d(net, self.w_norm(w, training=training), strides=[1, self.strides[0], self.strides[1], 1],
                                   dilations=self.atrous_rate, padding="VALID",
                                   data_format='NHWC'))
            net = tf.concat(outputs, axis=-1)
        else:
            if self.strides[0] == 2:
                w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
                w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
            net = tf.nn.conv2d(net, self.w_norm(w, training=training), strides=[1, self.strides[0], self.strides[1], 1],
                               dilations=self.atrous_rate, padding="VALID",
                               data_format='NHWC')
        return self.conv_aftermath(net, training=training)


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
    def __init__(self, kernel_size, strides=(1, 1), atrous_rate=1, use_bias=True, use_scale=True, norm=None,
                 w_norm=None, act=None, convolution_kernel=None, rotate_filters=False,
                 padding="same", padding_algorithm="REFLECT"):
        self.padding = padding.upper()
        self.padding_algorithm = padding_algorithm.upper()
        self.act = act
        self.norm = norm
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.strides = [1, strides[0], strides[1], 1]
        self.atrous_rate = [atrous_rate, atrous_rate]
        self.kernel_size = kernel_size
        self.weight = convolution_kernel
        self.rotate_filters = rotate_filters
        if self.weight is not None:
            self.kernel_size = self.weight.shape[0]

        self.virtual_kernel_size = (atrous_rate * (self.kernel_size - 1)) + 1
        if self.padding == "SAME":
            self.pad = Padding(ks=self.virtual_kernel_size, s=self.strides[0], algorithm=self.padding_algorithm)
        else:
            self.pad = lambda x: x

        self.conv_aftermath = ConvAftermath(use_bias=self.use_bias, use_scale=self.use_scale,
                                            norm=self.norm, act=self.act)
        if w_norm is not None:
            self.w_norm = w_norm()
        else:
            self.w_norm = lambda x, training=True: x
        super(DepthwiseConv2D, self).__init__()

    def build(self, input_shape):
        kernel_shape = (self.kernel_size, self.kernel_size, input_shape[-1], 1)
        if self.weight is None:
            self.weight = get_weight(shape=kernel_shape)
        super(DepthwiseConv2D, self).build(input_shape)

    def call(self, inputs, training=None):
        net = self.pad(inputs)
        w = self.weight
        if self.rotate_filters:
            w = kernel_rotation(w)
        net = tf.nn.depthwise_conv2d(net, filter=self.w_norm(w, training=training), strides=self.strides,
                                     padding="VALID",
                                     dilations=self.atrous_rate, data_format="NHWC")
        return self.conv_aftermath(net, training=training)
