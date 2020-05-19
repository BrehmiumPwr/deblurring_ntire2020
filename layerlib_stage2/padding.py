import tensorflow as tf


class Padding(tf.keras.layers.Layer):
    def __init__(self, ks, s=1, algorithm="REFLECT"):
        self.ks = ks
        self.s = s
        self.padding_algorithm = algorithm
        self.p = int((ks - 1) / 2)
        self.pad_start = [self.p, self.p]
        self.pad_end = [self.p, self.p]
        self.do_nothing = self.ks == 1 and self.s == 1
        super(Padding, self).__init__(trainable=False)


    def call(self, inputs):
        if self.do_nothing:
            return inputs

        if self.s > 1:
            #  (n+2*p-f)
            input_spatial_dims = tf.shape(inputs)[1:3]

            output_spatial_dims = tf.cast(tf.math.ceil(input_spatial_dims / self.s), dtype=tf.int32)
            padding = (output_spatial_dims * self.s + self.ks - 1) - input_spatial_dims
            pad_start = padding // 2
            pad_end = padding - pad_start
        else:
            pad_start = self.pad_start
            pad_end = self.pad_end

        return tf.pad(inputs, [[0, 0], [pad_start[0], pad_end[0]], [pad_start[1], pad_end[1]], [0, 0]], self.padding_algorithm)