import tensorflow as tf
import tensorflow.keras as keras
from .convolution import Conv2D, AtrousConv2D, AtrousConv2D_ReflectPad, Conv2D_ReflectPad, ChannelAttention, SpatialAttention


class AtrousBlockPad(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, use_bias, use_scale, activation, atrousBlocks=[1,2,3,4]):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.activation = activation
        self.needs_projection = max(self.strides) > 1

        self.atrous_layers = []
        for i in range(4):
            self.atrous_layers.append(AtrousConv2D_ReflectPad(filters=int(self.filters/4),
                                                              kernel_size=self.kernel_size,
                                                              strides=self.strides,
                                                              dilation=atrousBlocks[i],
                                                              use_bias=self.use_bias,
                                                              use_scale=self.use_scale,
                                                              act=self.activation))

        self.conv1 = Conv2D_ReflectPad(filters=self.filters,
                                       kernel_size=1,
                                       strides=self.strides,
                                       padding="same",
                                       use_bias=self.use_bias,
                                       use_scale=self.use_scale,
                                       act=self.activation)
        super(AtrousBlockPad, self).__init__()

    def build(self, input_shape):
        dims_match = input_shape[-1] != self.filters
        self.needs_projection = self.needs_projection or dims_match
        if self.needs_projection:
            self.projection = Conv2D_ReflectPad(filters=self.filters,
                                                kernel_size=1,
                                                strides=self.strides,
                                                padding="same",
                                                use_bias=self.use_bias,
                                                use_scale=self.use_scale,
                                                act=self.activation)

        super(AtrousBlockPad, self).build(input_shape)

    def call(self, input_data):
        x = input_data

        x1 = self.atrous_layers[0](x)
        x2 = self.atrous_layers[1](x)
        x3 = self.atrous_layers[2](x)
        x4 = self.atrous_layers[3](x)

        data = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
        data = self.conv1(data)
        if self.needs_projection:
            input_data = self.projection(input_data)
        return input_data + data


class AtrousBlockPad2(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, use_bias, use_scale, activation, atrousBlocks=[1,2,3,4]):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.activation = activation
        self.needs_projection = max(self.strides) > 1

        self.atrous_layers = []
        for i in range(4):
            self.atrous_layers.append(AtrousConv2D_ReflectPad(filters=int(self.filters/2),
                                                              kernel_size=self.kernel_size,
                                                              strides=self.strides,
                                                              dilation=atrousBlocks[i],
                                                              use_bias=self.use_bias,
                                                              use_scale=self.use_scale,
                                                              act=self.activation))

        self.conv1 = Conv2D_ReflectPad(filters=self.filters,
                                       kernel_size=self.kernel_size,
                                       strides=self.strides,
                                       padding="same",
                                       use_bias=self.use_bias,
                                       use_scale=self.use_scale,
                                       act=self.activation)
        super(AtrousBlockPad2, self).__init__()

    def build(self, input_shape):
        dims_match = input_shape[-1] != self.filters
        self.needs_projection = self.needs_projection or dims_match
        if self.needs_projection:
            self.projection = Conv2D_ReflectPad(filters=self.filters,
                                                kernel_size=1,
                                                strides=self.strides,
                                                padding="same",
                                                use_bias=self.use_bias,
                                                use_scale=self.use_scale,
                                                act=self.activation)

        super(AtrousBlockPad2, self).build(input_shape)

    def call(self, input_data):
        x = input_data

        x1 = self.atrous_layers[0](x)
        x2 = self.atrous_layers[1](x)
        x3 = self.atrous_layers[2](x)
        x4 = self.atrous_layers[3](x)

        data = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
        data = self.conv1(data)
        if self.needs_projection:
            input_data = self.projection(input_data)
        return input_data + data





class AtrousBlockPadAttention(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, use_bias, use_scale, activation, atrousBlocks=[1,2,3,4]):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.activation = activation
        self.needs_projection = max(self.strides) > 1

        self.atrous_layers = []
        for i in range(4):
            self.atrous_layers.append(AtrousConv2D_ReflectPad(filters=int(self.filters/2),
                                                              kernel_size=self.kernel_size,
                                                              strides=self.strides,
                                                              dilation=atrousBlocks[i],
                                                              use_bias=self.use_bias,
                                                              use_scale=self.use_scale,
                                                              act=self.activation))

        self.attentionLayerCh = ChannelAttention(ratio=16)
        self.attentionLayerSp = SpatialAttention(kernel_size=7)

        self.conv1 = Conv2D_ReflectPad(filters=self.filters,
                                       kernel_size=self.kernel_size,
                                       strides=self.strides,
                                       padding="same",
                                       use_bias=self.use_bias,
                                       use_scale=self.use_scale,
                                       act=self.activation)
        super(AtrousBlockPadAttention, self).__init__()

    def build(self, input_shape):
        dims_match = input_shape[-1] != self.filters
        self.needs_projection = self.needs_projection or dims_match
        if self.needs_projection:
            self.projection = Conv2D_ReflectPad(filters=self.filters,
                                                kernel_size=1,
                                                strides=self.strides,
                                                padding="same",
                                                use_bias=self.use_bias,
                                                use_scale=self.use_scale,
                                                act=self.activation)

        super(AtrousBlockPadAttention, self).build(input_shape)

    def call(self, input_data):
        x = input_data

        x1 = self.atrous_layers[0](x)
        x2 = self.atrous_layers[1](x)
        x3 = self.atrous_layers[2](x)
        x4 = self.atrous_layers[3](x)
        data = tf.keras.layers.Concatenate()([x1, x2, x3, x4])

        data = self.attentionLayerCh(data)
        data = self.attentionLayerSp(data)

        data = self.conv1(data)
        if self.needs_projection:
            input_data = self.projection(input_data)
        return input_data + data





