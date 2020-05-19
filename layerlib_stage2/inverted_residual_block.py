import tensorflow as tf
import tensorflow.keras as keras
from . import Conv2D, DepthwiseConv2D


class InvertedResidualBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, use_bias, use_scale, activation, rotate_filters=False, norm=None, w_norm=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.expansion = 6
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.activation = activation
        self.norm = norm
        self.w_norm = w_norm
        self.rotate_filters = rotate_filters
        self.needs_projection = min(self.strides) > 2
        super(InvertedResidualBlock, self).__init__()

    def build(self, input_shape):
        dims_match = input_shape[-1] != self.filters
        intermediate_dim = input_shape[-1] * self.expansion

        self.conv1 = Conv2D(filters=intermediate_dim, kernel_size=1, padding="same", norm=self.norm, w_norm=self.w_norm)
        self.conv2 = DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.strides, padding="same", norm=self.norm,
                                     w_norm=self.w_norm, rotate_filters=self.rotate_filters)
        self.conv3 = Conv2D(filters=self.filters, kernel_size=1, padding="same", norm=self.norm, w_norm=self.w_norm)

        self.needs_projection = self.needs_projection or dims_match
        if self.needs_projection:
            self.projection = Conv2D(filters=self.filters,
                                     kernel_size=1,
                                     strides=self.strides,
                                     padding="same",
                                     use_bias=self.use_bias,
                                     use_scale=self.use_scale,
                                     act=self.activation,
                                     norm=self.norm,
                                     w_norm=self.w_norm)

        super(InvertedResidualBlock, self).build(input_shape)

    def call(self, input_data, training=None):
        data = self.conv1(input_data, training=training)
        data = self.conv2(data, training=training)
        data = self.conv3(data, training=training)
        if self.needs_projection:
            input_data = self.projection(input_data, training=training)
        return input_data + data
