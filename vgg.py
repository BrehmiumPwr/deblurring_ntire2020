import tensorflow as tf

class VGG16Loss(object):

    def __init__(self):
        self.vgg16 = tf.keras.applications.VGG16(include_top=False, weights="imagenet", pooling=None)
        self.conv33 = self.vgg16.get_layer("block3_conv3").output
        self.vgg_loss_model = tf.keras.models.Model([self.vgg16.input], self.conv33)


    def __call__(self, x_real, x_fake):
        x_real *= 127.5
        x_fake *= 127.5
        y_real = self.vgg_loss_model(x_real, training=False)
        y_fake = self.vgg_loss_model(x_fake, training=False)
        return tf.reduce_mean(tf.losses.mean_absolute_error(y_true=y_real, y_pred=y_fake)) / 1000.0