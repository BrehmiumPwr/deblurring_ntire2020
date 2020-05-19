import tensorflow as tf
import tensorflow.keras as keras
from layerlib_stage2.convolution import Conv2D, DepthwiseConv2D


class SeparableResidualBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, use_bias, use_scale, activation, rotate_filters=False, norm=None, w_norm=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.activation = activation
        self.norm = norm
        self.w_norm = w_norm
        self.needs_projection = max(self.strides) > 1

        self.conv11 = DepthwiseConv2D(self.kernel_size,
                                      strides=self.strides,
                                      use_bias=self.use_bias,
                                      use_scale=self.use_scale,
                                      norm=self.norm,
                                      w_norm=None,
                                      act=None,
                                      rotate_filters=rotate_filters,
                                      padding="same",
                                      padding_algorithm="REFLECT")
        self.conv12 = Conv2D(filters=self.filters,
                             kernel_size=1,
                             strides=(1, 1),
                             padding="same",
                             use_bias=self.use_bias,
                             use_scale=self.use_scale,
                             norm=self.norm,
                             w_norm=self.w_norm,
                             act=self.activation,
                             padding_algorithm="REFLECT")
        self.conv21 = DepthwiseConv2D(self.kernel_size,
                                      strides=(1, 1),
                                      use_bias=self.use_bias,
                                      use_scale=self.use_scale,
                                      norm=self.norm,
                                      w_norm=None,
                                      act=None,
                                      padding="same",
                                      padding_algorithm="REFLECT")
        self.conv22 = Conv2D(filters=self.filters,
                             kernel_size=1,
                             strides=(1, 1),
                             padding="same",
                             use_bias=self.use_bias,
                             use_scale=self.use_scale,
                             norm=self.norm,
                             w_norm=self.w_norm,
                             act=None,
                             padding_algorithm="REFLECT")
        super(SeparableResidualBlock, self).__init__()

    def build(self, input_shape):
        dims_match = input_shape[-1] == self.filters
        self.needs_projection = self.needs_projection or not dims_match
        if self.needs_projection:
            self.projection = Conv2D(filters=self.filters,
                                     kernel_size=1,
                                     strides=self.strides,
                                     padding="same",
                                     use_bias=self.use_bias,
                                     use_scale=self.use_scale,
                                     norm=self.norm,
                                     w_norm=self.w_norm,
                                     act=None)

        super(SeparableResidualBlock, self).build(input_shape)

    def call(self, input_data, training=None):
        data = self.activation(input_data)
        data = self.conv11(data, training=training)
        data = self.conv12(data, training=training)
        data = self.conv21(data, training=training)
        data = self.conv22(data, training=training)
        if self.needs_projection:
            input_data = self.projection(input_data, training=training)
        return input_data + data
