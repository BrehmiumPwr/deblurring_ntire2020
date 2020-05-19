import tensorflow as tf


class PixelNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        super(PixelNorm, self).__init__()

    def call(self, inputs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon)