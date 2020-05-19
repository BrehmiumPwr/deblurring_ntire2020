import tensorflow as tf
import tensorflow.keras as keras
from layerlib_stage2 import Conv2D, Conv2DTranspose, OffsetConvolution, ResidualBlock, InvertedResidualBlock
import tensorflow_addons as tfa


class VideoBasenet(keras.layers.Layer):
    def __init__(self, num_blocks=3, max_global_stride=16, pad_to_fit_global_stride=True, d_mult=16, time_steps=1,
                 activation=tf.nn.relu, block_type=InvertedResidualBlock, feature_normalization=None,
                 weight_normalization=None):
        self.num_blocks = num_blocks
        self.max_global_stride = max_global_stride
        self.pad_to_fit_global_stride = pad_to_fit_global_stride
        self.d_mult = d_mult
        self.d_blocks = self.d_mult * self.max_global_stride
        self.activation = activation
        self.feature_normalization = feature_normalization
        self.weight_normalization = weight_normalization
        self.time_steps = time_steps
        self.center_idx = self.time_steps // 2
        self.downsampling_layers = []
        cur_global_stride = 1
        self.downsampling_layers.append(Conv2D(filters=self.d_mult // 4,
                                               kernel_size=7,
                                               strides=(1, 1),
                                               use_bias=True,
                                               use_scale=True,
                                               padding="same",
                                               norm=None,
                                               rotate_filters=True,
                                               w_norm=self.weight_normalization,
                                               act=self.activation))
        while cur_global_stride < max_global_stride:
            self.downsampling_layers.append(
                Conv2D(filters=cur_global_stride * self.d_mult,
                       kernel_size=3,
                       strides=(2, 2),
                       use_bias=True,
                       use_scale=True,
                       padding="same",
                       rotate_filters=False,
                       norm=self.feature_normalization,
                       w_norm=self.weight_normalization,
                       act=self.activation,
                       dropout=0.0))
            cur_global_stride *= 2

        self.fusion_layers = []
        self.upsampling_layers = []
        self.normalization_layers = []
        self.blocks = []
        while cur_global_stride > 1:
            cur_global_stride /= 2
            self.fusion_layers.append(Conv2D(filters=int(cur_global_stride * self.d_mult)*2,
                                             kernel_size=3,
                                             strides=(1, 1),
                                             use_bias=True,
                                             use_scale=True,
                                             padding="same",
                                             rotate_filters=False,
                                             norm=self.feature_normalization,
                                             w_norm=self.weight_normalization,
                                             act=self.activation,
                                             dropout=0.0))
            cur_blocks = []
            for x in range(num_blocks):
                cur_blocks.append(block_type(filters=int(cur_global_stride * self.d_mult),
                                             kernel_size=3,
                                             strides=(1, 1),
                                             use_bias=True,
                                             use_scale=True,
                                             norm=self.feature_normalization,
                                             w_norm=self.weight_normalization,
                                             activation=self.activation,
                                             dropout=0.0))
            self.blocks.append(cur_blocks)
            self.normalization_layers.append(tfa.layers.normalizations.InstanceNormalization())
            self.upsampling_layers.append(Conv2DTranspose(filters=int(cur_global_stride * self.d_mult),
                                                          kernel_size=3,
                                                          strides=(2, 2),
                                                          padding="same",
                                                          act=self.activation,
                                                          use_bias=True,
                                                          use_scale=True,
                                                          norm=self.feature_normalization,
                                                          w_norm=self.weight_normalization
                                                          ))

        self.output_layer = Conv2D(filters=3,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   use_bias=True,
                                   use_scale=True,
                                   padding="same",
                                   norm=None,
                                   w_norm=self.weight_normalization,
                                   act=None)

        super(VideoBasenet, self).__init__()

    def flatten_sequence_dim(self, batch):
        # get rid of sequence dimension by incorporating it into the channel dim
        images = tf.transpose(batch, perm=[0, 2, 3, 4, 1])
        images = tf.reshape(images, shape=tf.concat([tf.shape(images)[:-2], [-1]], axis=0))
        return images

    def call(self, input_data, training=None):

        input_shape = tf.shape(input_data)
        # if input_shape[1] == 1:
        #    # single mode
        #    input_data = self.flatten_sequence_dim(input_data)

        net = input_data
        # select center frame for residual
        center_frame = input_data[:, input_shape[1] // 2, :, :, :]

        def downsampling(features_frame):
            downs = []
            for x in range(len(self.downsampling_layers)):
                features_frame = self.downsampling_layers[x](features_frame, training=training)
                downs.append(features_frame)
            return downs

        frames = tf.transpose(net, perm=[1, 0, 2, 3, 4])

        downs = []
        for x in range(self.time_steps):
            features = downsampling(frames[x])
            downs.append(features)

        # features = [x[-1] for x in downs]
        # net = tf.concat(features, axis=-1)

        def similarity(reference, others):
            similarities = []
            for x in others:
                sim = reference * x
                sim = tf.reduce_sum(sim, axis=[-1], keepdims=True)
                similarities.append(sim)
            return similarities

        for x in range(len(self.upsampling_layers)):
            idx = len(downs[self.center_idx]) - x - 1

            multi_stream_features = [downs[y][idx] for y in range(len(downs))]
            multi_stream_features = [tf.nn.l2_normalize(feat, axis=-1) for feat in multi_stream_features]
            reference_features = multi_stream_features[self.center_idx]
            similarities = similarity(reference_features, multi_stream_features)

            multi_stream_features = [multi_stream_features[c] * similarities[c] for c in
                                     range(len(multi_stream_features))]
            #if x > 0:
            #    net = tf.concat([net, *multi_stream_features], axis=-1)
            #else:
            msfeat = tf.concat(multi_stream_features, axis=-1)
            msfeat = self.fusion_layers[x](msfeat, training=training)
            if x > 0:
                net =  msfeat + net
            else:
                net = msfeat
            for z in range(len(self.blocks[x])):
                net = self.blocks[x][z](net, training=training)
            # probably need to do some normalization here
            # net = self.normalization_layers[x](net)
            net = self.upsampling_layers[x](net, training=training)

        net += downs[self.center_idx][0]

        # might be a good idea to concat level 0 features of the reference framehere for better alignment to edges
        net = self.output_layer(net, training=training)
        # weight_skip = tf.nn.relu(net[:, :, :, 3:4])
        # weight_net = tf.nn.relu(net[:, :, :, 4:5])
        # net = net[:, :, :, :3]
        # return (input_data * weight_skip) + (net * weight_net)
        return net + center_frame, 0.0
