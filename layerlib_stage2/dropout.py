import tensorflow as tf

class Dropout(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.0):
        self.dropout_rate = dropout_rate
        super(Dropout, self).__init__()

    def build(self, input_shape):
        channels = input_shape[-1]
        batch_size = input_shape[0]
        noise_shape = [batch_size, 1, 1, channels]
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, noise_shape=noise_shape)

    def call(self, inputs, training=None):
        return self.dropout(inputs, training=training)