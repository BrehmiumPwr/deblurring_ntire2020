
import tensorflow as tf

class SpectralNorm(tf.keras.layers.Layer):
    def __init__(self, iterations=1):
        self.u = None
        self.shape = None
        self.iterations = iterations
        self.initializer = tf.keras.initializers.RandomNormal()
        super(SpectralNorm, self).__init__()

    def build(self, input_shape):
        self.shape = input_shape
        self.u = tf.Variable(trainable=False, initial_value=self.initializer(shape=[1, input_shape[-1]], dtype=tf.float32))
        super(SpectralNorm, self).build(input_shape)

    def call(self, inputs, training=True):
        w = tf.reshape(inputs, [-1, self.shape[-1]])
        u_hat = self.u
        v_hat = None
        for i in range(self.iterations):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        if training:
            with tf.control_dependencies([self.u.assign(u_hat)]):
                w_norm = w / sigma
                w_norm = tf.reshape(w_norm, self.shape)
        else:
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, self.shape)
        return w_norm


class UnitNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(UnitNorm, self).__init__()

    def call(self, inputs):
        # assume weights have shape [ks, ks, input_maps, filters]
        w_squared = tf.square(inputs) + 1e-8
        w_sum = tf.sqrt(tf.reduce_sum(w_squared, axis= [0,1,2], keepdims=True))
        return inputs / w_sum

