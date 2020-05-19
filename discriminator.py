import tensorflow as tf
import tensorflow.keras as keras
from layerlib_stage2 import Conv2D, ResidualBlock, InvertedResidualBlock


class ScalarFusionNetwork(keras.layers.Layer):
    def __init__(self, num_layers, num_output_features, activation, weight_normalization=None):
        self.num_layers = num_layers
        self.num_output_features = num_output_features
        self.activation = activation
        self.weight_normalization = weight_normalization
        self.layers = []
        for x in range(self.num_layers):
            self.layers.append(
                Conv2D(filters=num_output_features,
                       kernel_size=1,
                       strides=(1, 1),
                       use_bias=True,
                       use_scale=True,
                       padding="same",
                       norm=None,
                       w_norm=self.weight_normalization,
                       act=self.activation)
            )
        super(ScalarFusionNetwork, self).__init__()

    def call(self, inputs):
        for x in range(self.num_layers):
            inputs = self.layers[x](inputs)
        return inputs


class Discriminator(keras.layers.Layer):
    def __init__(self, max_global_stride=8, pad_to_fit_global_stride=True, d_mult=16,
                 activation=tf.nn.relu, block_type=InvertedResidualBlock, feature_normalization=None,
                 weight_normalization=None):
        self.max_global_stride = max_global_stride
        self.pad_to_fit_global_stride = pad_to_fit_global_stride
        self.d_mult = d_mult
        self.activation = activation
        self.feature_normalization = feature_normalization
        self.weight_normalization = weight_normalization
        self.scalar_fusion_network = ScalarFusionNetwork(5, 256, activation=self.activation)

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

        self.fusion_layers = []
        for idx in range(len(self.layers)-1):
            self.fusion_layers.append(
                Conv2D(filters=self.layers[idx].filters*2,
                       kernel_size=1,
                       strides=(1, 1),
                       use_bias=True,
                       use_scale=True,
                       padding="same",
                       norm=None,
                       w_norm=self.weight_normalization,
                       act=None))
        super(Discriminator, self).__init__()

    def call(self, fake_data, real_data, training=None):
        stddev_real = tf.math.reduce_std(real_data, axis=[1,2,3], keepdims=True)
        stddev_fake = tf.math.reduce_std(fake_data, axis=[1,2,3], keepdims=True)

        stddevs = tf.concat([stddev_fake, stddev_real], axis=-1)
        fusion_features = self.scalar_fusion_network(stddevs)

        net = tf.concat([real_data, fake_data], axis=-1)
        for x in range(len(self.layers)-1):
            net = self.layers[x](net, training=training)
            net *= self.fusion_layers[x](fusion_features)[:self.layers[x].filters]
            net += self.fusion_layers[x](fusion_features)[self.layers[x].filters:]

        net = self.layers[-1](net, training=training)
        return net
