import tensorflow as tf
from gan import StandardGAN, RelativisticGAN, RelativisticAvgGAN
from layerlib_stage2 import ResidualBlock, InvertedResidualBlock, SeparableResidualBlock, AtrousResidualBlock, \
    DepthwiseAtrousResidualBlock, feature_normalization, weight_normalization
from layerlib_stage2 import DepthwiseConv2D
from video_basenet4 import VideoBasenet
from discriminators import discriminators
from vgg import VGG16Loss
import numpy as np
import tensorflow_addons as tfa

model_factory = {
    "video_basenet": VideoBasenet,
}


def exponential_decay(initial_learning_rate, decay_rate, step, decay_steps):
    return tf.pow(initial_learning_rate * decay_rate,
                  tf.cast(step, dtype=tf.float32) + 1e-8 / (tf.cast(decay_steps, dtype=tf.float32)))


class DeblurModel(object):
    def __init__(self, use_gan=True, use_reconstruction=True, use_vgg=True,
                 hard_example_mining=True, delta_learning=True, num_steps_video=1,
                 max_global_stride=4, gen_type="video_basenet", disc_type="discriminator"):
        self.use_gan = use_gan
        self.use_reconstruction = use_reconstruction
        self.use_vgg = use_vgg
        self.hard_example_mining = hard_example_mining
        self.delta_learning = delta_learning
        self.gan_type = "relativistic"
        self.summary_steps = 100
        self.disc = discriminators[disc_type]
        self.num_steps_video = num_steps_video
        self.max_global_stride = max_global_stride
        # super(DeblurModel, self).__init__()
        self.network = model_factory[gen_type](num_blocks=3,
                                               max_global_stride=self.max_global_stride,
                                               pad_to_fit_global_stride=True,
                                               d_mult=64,
                                               activation=tf.nn.leaky_relu,
                                               block_type=ResidualBlock,
                                               time_steps=self.num_steps_video,
                                               # feature_normalization=feature_normalization.PixelNorm,
                                               weight_normalization=weight_normalization.UnitNorm
                                               )
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            0.0001,
            decay_steps=100000,
            decay_rate=0.5,
            staircase=True)

        self.learning_rate = 0.001
        self.step = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.optimizer_g = tf.optimizers.Adam(learning_rate=self.learning_rate)

        if self.use_gan:
            self.discriminator = self.disc(max_global_stride=4,
                                           pad_to_fit_global_stride=True,
                                           d_mult=64,
                                           activation=tf.nn.leaky_relu,
                                           block_type=ResidualBlock,
                                           # feature_normalization=feature_normalization.PixelNorm,
                                           weight_normalization=weight_normalization.UnitNorm
                                           )

            gan_types = {
                "sgan": StandardGAN(
                    reduction=lambda x: tf.reduce_mean(tf.nn.top_k(tf.reshape(x, shape=[-1]), k=80)[0])),
                "relativistic": RelativisticAvgGAN(type="lsgan")
            }
            self.gan = gan_types[self.gan_type]
            self.gan_loss_g = self.gan.generator
            self.gan_loss_d = self.gan.discriminator
            self.gan_weight = 1.0
            self.optimizer_d = tf.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            self.discriminator = lambda x: x
            self.gan_loss = lambda x, label: 0
            self.gan_weight = 0

        if self.use_vgg:
            self.vgg = VGG16Loss()

    def checkpoint(self):
        if self.use_gan:
            return tf.train.Checkpoint(optimizer_d=self.optimizer_d,
                                       optimizer_g=self.optimizer_g,
                                       generator=self.network,
                                       discriminator=self.discriminator)
        else:
            return tf.train.Checkpoint(optimizer_g=self.optimizer_g,
                                       generator=self.network)

    def load(self, ckpt_path, expect_partial=False):
        ckpt = self.checkpoint()
        ckpt_filename = tf.train.latest_checkpoint(checkpoint_dir=ckpt_path)
        ckpt = ckpt.restore(ckpt_filename)
        if expect_partial:
            ckpt.expect_partial()
        print(" [*] Loading model from {}".format(ckpt_filename), flush=True)

    def discriminator_step(self, images_blurred, images_sharp):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.discriminator.trainable_variables + [images_sharp])
            output = self.network(images_blurred, training=False)
            fake_scores = self.discriminator(output, images_sharp, training=True)
            real_scores = self.discriminator(images_sharp, images_sharp, training=True)
            d_loss_gan = self.gan_loss_d(real_scores=real_scores, fake_scores=fake_scores)
            d_reg_loss = 0  # discriminator_regularizer(images_sharp, real_scores, tape=tape)
            d_loss = d_loss_gan + d_reg_loss
            if self.step % self.summary_steps == 0:
                tf.summary.scalar(name="loss/discriminator_gan", data=d_loss_gan, step=self.step)
                tf.summary.scalar(name="loss/discriminator_reg", data=d_reg_loss, step=self.step)
                tf.summary.scalar(name="loss/discriminator_total", data=d_loss, step=self.step)

        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        del tape
        gradients = [tf.clip_by_norm(grad, 5) for grad in gradients]
        if self.step % self.summary_steps == 0:
            for x in range(len(gradients)):
                tf.summary.histogram(name="gradients/discriminator/" + self.discriminator.trainable_variables[x].name,
                                     data=gradients[x], step=self.step)
                tf.summary.scalar(
                    name="gradients/magnitude/discriminator/" + self.discriminator.trainable_variables[x].name,
                    data=tf.reduce_mean(tf.abs(gradients[x])), step=self.step)
        self.optimizer_d.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        return d_loss

    def generator_step(self, images_blurred, images_sharp):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            g_loss = 0
            output, surround_loss = self.network(images_blurred, training=True)
            g_reconstruction_loss = 0
            if isinstance(output, list):
                image_size = tf.shape(images_sharp)[1:3]
                resized_sharp_images = [tf.image.resize(images_sharp,
                                                        size=tf.shape(out)[1:3],
                                                        method=tf.image.ResizeMethod.BILINEAR)
                                        for out in output]
                resized_sharp_images[-1] = images_sharp
                for x in range(len(output)):
                    cur_out = output[x]
                    g_reconstruction_loss += tf.reduce_mean(tf.abs(cur_out - resized_sharp_images[x]))
            else:
                # reconstruction loss
                tf.shape(images_sharp)
                g_reconstruction_loss += tf.abs(output - images_sharp)  # + moment_loss
            mean_error = tf.reduce_mean(g_reconstruction_loss)
            if self.hard_example_mining:
                error_threshold = 1.0
                weight = tf.cast(tf.greater(g_reconstruction_loss, mean_error * error_threshold), dtype=tf.int32)
                num_values = tf.reduce_sum(weight)
                g_reconstruction_loss = tf.reduce_sum(g_reconstruction_loss * tf.cast(weight, dtype=tf.float32))
                g_reconstruction_loss /= tf.cast(num_values, dtype=tf.float32)
            else:
                g_reconstruction_loss = mean_error + surround_loss

            if self.use_reconstruction:
                g_loss += g_reconstruction_loss

            if self.use_gan:
                fake_scores = self.discriminator(output, images_sharp, training=False)
                real_scores = self.discriminator(images_sharp, images_sharp, training=False)
                g_loss_gan = self.gan_loss_g(fake_scores=fake_scores, real_scores=real_scores)
            else:
                g_loss_gan = 0.0
            g_loss += (self.gan_weight * g_loss_gan)

            vgg_loss = 0.0
            if self.use_vgg:
                if isinstance(output, list):
                    for x in range(len(output)):
                        cur_out = output[x]
                        vgg_loss += self.vgg(x_real=resized_sharp_images[x], x_fake=cur_out)
                else:
                    vgg_loss += self.vgg(x_real=images_sharp, x_fake=output)

            g_loss += vgg_loss

            if self.step % self.summary_steps == 0:
                tf.summary.scalar(name="loss/generator_gan", data=g_loss_gan, step=self.step)
                tf.summary.scalar(name="loss/generator_vgg", data=vgg_loss, step=self.step)
                tf.summary.scalar(name="loss/generator_reconstruction", data=g_reconstruction_loss, step=self.step)
                tf.summary.scalar(name="loss/generator_total", data=g_loss, step=self.step)
        gradients = tape.gradient(g_loss, self.network.trainable_variables)
        gradients = [tf.clip_by_norm(grad, 5) for grad in gradients]
        self.optimizer_g.apply_gradients(zip(gradients, self.network.trainable_variables))
        return g_loss

    @tf.function
    def train_step(self, images_blurred, images_sharp):
        tf.summary.experimental.set_step(self.step)
        g_loss = self.generator_step(images_blurred, images_sharp)
        if self.use_gan:
            d_loss = self.discriminator_step(images_blurred, images_sharp)
        else:
            d_loss = 0.0
        if self.step % self.summary_steps == 0:
            tf.summary.scalar(name="general/learning_rate", data=self.learning_rate, step=self.step)
        self.step.assign_add(delta=1)
        return d_loss, g_loss

    def build(self, input_shape):
        pass

    def __call__(self, inputs):
        single_image = False
        if len(inputs.shape) == 3:
            # add batch dimension
            single_image = True
            inputs = tf.expand_dims(inputs, axis=0)
            # add sequence dimension
            inputs = tf.expand_dims(inputs, axis=1)
        inputs = tf.cast(inputs, dtype=tf.float32)
        inputs /= 127.5
        inputs -= 1.0
        output = self.network(inputs, training=False)
        output = tf.clip_by_value(output, clip_value_min=-1.0, clip_value_max=1.0)
        output += 1.0
        output *= 127.5

        # remove batch dimension
        output = output[0]
        return output
