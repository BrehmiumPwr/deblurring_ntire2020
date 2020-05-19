import tensorflow as tf
import tensorflow.keras as keras
from layerlib_stage2.convolution import Conv2D, OffsetConvolution
from .dropout import Dropout

class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, use_bias, use_scale, activation, conv_type=Conv2D,
                 rotate_filters=False, norm=None, w_norm=None, dropout=0.0):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.activation = activation
        self.norm = norm
        self.w_norm = w_norm
        self.conv_type = conv_type
        self.dropout_rate = dropout
        self.needs_projection = max(self.strides) > 1
        self.conv1 = self.conv_type(filters=self.filters if not rotate_filters else self.filters // 4,
                                    kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding="same",
                                    use_bias=self.use_bias,
                                    use_scale=self.use_scale,
                                    norm=self.norm,
                                    w_norm=self.w_norm,
                                    rotate_filters=rotate_filters,
                                    act=self.activation,
                                    dropout=0.0)
        self.conv2 = self.conv_type(filters=self.filters,
                                    kernel_size=self.kernel_size,
                                    strides=(1, 1),
                                    padding="same",
                                    use_bias=self.use_bias,
                                    use_scale=self.use_scale,
                                    norm=self.norm,
                                    w_norm=self.w_norm,
                                    act=None,
                                    dropout=0.0)
        if self.dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate=self.dropout_rate)

        super(ResidualBlock, self).__init__()

    def build(self, input_shape):
        if self.conv_type == OffsetConvolution:
            filters = self.filters * 5
        else:
            filters = self.filters
        dims_match = input_shape[-1] == filters
        self.needs_projection = self.needs_projection or not dims_match
        if self.needs_projection:
            self.projection = self.conv_type(filters=filters,
                                             kernel_size=1,
                                             strides=self.strides,
                                             padding="same",
                                             use_bias=self.use_bias,
                                             use_scale=self.use_scale,
                                             norm=self.norm,
                                             w_norm=self.w_norm,
                                             act=None)

        super(ResidualBlock, self).build(input_shape)

    def call(self, input_data, training=None):
        data = self.activation(input_data)
        data = self.conv1(data, training=training)
        data = self.conv2(data, training=training)
        if self.needs_projection:
            input_data = self.projection(input_data, training=training)
        net = input_data + data
        if self.dropout_rate > 0.0:
            net = self.dropout(net, training=training)
        return net
