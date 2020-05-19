import tensorflow as tf
import tensorflow.keras as keras
from layerlib_stage2 import Conv2D, ResidualBlock, InvertedResidualBlock, DepthwiseConv2D
import numpy as np

class EdgeDiscriminator(keras.layers.Layer):
    def __init__(self, max_global_stride=8, pad_to_fit_global_stride=True, d_mult=16,
                 activation=tf.nn.relu, block_type=InvertedResidualBlock, feature_normalization=None,
                 weight_normalization=None):
        self.max_global_stride = max_global_stride
        self.pad_to_fit_global_stride = pad_to_fit_global_stride
        self.d_mult = d_mult
        self.activation = activation
        self.feature_normalization = feature_normalization
        self.weight_normalization = weight_normalization

        self.laplace_filter = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
        self.sobel_filter_x = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        self.sobel_filter_y = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]

        self.laplace_filter = np.stack([self.laplace_filter, self.laplace_filter, self.laplace_filter], axis=-1)
        self.sobel_filter_x = np.stack([self.sobel_filter_x, self.sobel_filter_x, self.sobel_filter_x], axis=-1)
        self.sobel_filter_y = np.stack([self.sobel_filter_y, self.sobel_filter_y, self.sobel_filter_y], axis=-1)

        self.laplace_filter = tf.constant(np.expand_dims(self.laplace_filter, axis=-1), dtype=tf.float32)
        self.sobel_filter_x = tf.constant(np.expand_dims(self.sobel_filter_x, axis=-1), dtype=tf.float32)
        self.sobel_filter_y = tf.constant(np.expand_dims(self.sobel_filter_y, axis=-1), dtype=tf.float32)

        self.laplace1 = DepthwiseConv2D(kernel_size=3, use_bias=False, use_scale=False, act=None,
                                       convolution_kernel=self.laplace_filter)
        self.sobel_x1 = DepthwiseConv2D(kernel_size=3, use_bias=False, use_scale=False, act=None,
                                       convolution_kernel=self.sobel_filter_x)
        self.sobel_y1 = DepthwiseConv2D(kernel_size=3, use_bias=False, use_scale=False, act=None,
                                       convolution_kernel=self.sobel_filter_y)
        self.laplace2 = DepthwiseConv2D(kernel_size=3, use_bias=False, use_scale=False, act=None,
                                       convolution_kernel=self.laplace_filter, atrous_rate=4)
        self.sobel_x2 = DepthwiseConv2D(kernel_size=3, use_bias=False, use_scale=False, act=None,
                                       convolution_kernel=self.sobel_filter_x, atrous_rate=4)
        self.sobel_y2 = DepthwiseConv2D(kernel_size=3, use_bias=False, use_scale=False, act=None,
                                       convolution_kernel=self.sobel_filter_y, atrous_rate=4)
        self.laplace3 = DepthwiseConv2D(kernel_size=3, use_bias=False, use_scale=False, act=None,
                                       convolution_kernel=self.laplace_filter, atrous_rate=8)
        self.sobel_x3 = DepthwiseConv2D(kernel_size=3, use_bias=False, use_scale=False, act=None,
                                       convolution_kernel=self.sobel_filter_x, atrous_rate=8)
        self.sobel_y3 = DepthwiseConv2D(kernel_size=3, use_bias=False, use_scale=False, act=None,
                                       convolution_kernel=self.sobel_filter_y, atrous_rate=8)
        self.layers = []
        self.layers.append(Conv2D(filters=self.d_mult,
                                  kernel_size=3,
                                  strides=(1, 1),
                                  use_bias=True,
                                  use_scale=True,
                                  padding="same",
                                  norm=None,
                                  w_norm=self.weight_normalization,
                                  act=self.activation))
        cur_global_stride = 1
        while cur_global_stride < max_global_stride:
            self.layers.append(block_type(filters=self.d_mult * cur_global_stride,
                                          kernel_size=3,
                                          strides=(2, 2),
                                          use_bias=True,
                                          use_scale=True,
                                          norm=self.feature_normalization,
                                          w_norm=self.weight_normalization,
                                          activation=self.activation))
            self.layers.append(block_type(filters=self.d_mult * cur_global_stride,
                                          kernel_size=3,
                                          strides=(1, 1),
                                          use_bias=True,
                                          use_scale=True,
                                          activation=self.activation))
            cur_global_stride *= 2

        self.layers.append(Conv2D(filters=1,
                                  kernel_size=3,
                                  strides=(1, 1),
                                  use_bias=True,
                                  use_scale=True,
                                  padding="same",
                                  w_norm=self.weight_normalization,
                                  norm=None,
                                  act=None))
        super(EdgeDiscriminator, self).__init__()


    def extract_edges_old(self, input):
        def single_scale(input, laplace_fn, sobel_x_fn, sobel_y_fn):
            laplace = laplace_fn(input)
            sobel_x = sobel_x_fn(input)
            sobel_y = sobel_y_fn(input)
            magnitude = tf.abs(sobel_x) + tf.abs(sobel_y)
            return tf.concat([laplace, sobel_x, sobel_y, magnitude], axis=-1)
        scale1 = single_scale(input, laplace_fn=self.laplace1, sobel_x_fn=self.sobel_x1, sobel_y_fn=self.sobel_y1)
        scale2 = single_scale(input, laplace_fn=self.laplace2, sobel_x_fn=self.sobel_x2, sobel_y_fn=self.sobel_y2)
        scale3 = single_scale(input, laplace_fn=self.laplace3, sobel_x_fn=self.sobel_x3, sobel_y_fn=self.sobel_y3)
        return tf.concat([scale1, scale2, scale3], axis=-1)

    def extract_edges(self, input):
        def single_scale(input):
            laplace = self.laplace1(input)
            sobel_x = self.sobel_x1(input)
            sobel_y = self.sobel_y1(input)
            magnitude = tf.abs(sobel_x) + tf.abs(sobel_y)
            return tf.concat([laplace, sobel_x, sobel_y, magnitude], axis=-1)
        scale1 = single_scale(input)
        input_shape = tf.shape(scale1)
        spatial_dims_scale1 = input_shape[1:3]
        spatial_dims_scale2 = spatial_dims_scale1 // 4
        spatial_dims_scale3 = spatial_dims_scale1 // 8
        input_scale2 = tf.image.resize(input, size=spatial_dims_scale2)
        input_scale3 = tf.image.resize(input, size=spatial_dims_scale3)

        scale2 = single_scale(input_scale2)
        scale2 = tf.image.resize(scale2, size=spatial_dims_scale1)
        scale3 = single_scale(input_scale3)
        scale3 = tf.image.resize(scale3, size=spatial_dims_scale1)
        return tf.concat([scale1, scale2, scale3], axis=-1)

    def call(self, input_data):
        net = input_data
        # extract edges
        net = self.extract_edges(net)
        for x in range(len(self.layers)):
            net = self.layers[x](net)
        return net
