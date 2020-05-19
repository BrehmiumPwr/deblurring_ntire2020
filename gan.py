import tensorflow as tf

def discriminator_regularizer(x_real, score_real, tape, act=tf.nn.sigmoid,  gamma=10.0):
    score_real = act(score_real)
    gradients_x = tape.gradient(score_real, x_real)
    disc_reg = gamma * 0.5 * tf.reduce_mean(tf.square(gradients_x))
    #tf.summary.scalar("disc_regularization", disc_reg)
    return disc_reg

class StandardGAN(tf.keras.layers.Layer):
    def __init__(self, reduction=tf.reduce_mean):
        self.reduction = reduction
        super(StandardGAN, self).__init__()

    def one_sided(self, scores, label):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(scores)*label, logits=scores)
        loss = self.reduction(loss)
        return loss

    def generator(self, real_scores, fake_scores):
        return self.one_sided(fake_scores, 1)

    def discriminator(self, real_scores, fake_scores):
        return self.one_sided(real_scores, 1) + self.one_sided(fake_scores, 0)


class RelativisticGAN(tf.keras.layers.Layer):
    def __init__(self, type="sgan"):
        if type in ["sgan"]:
            self.error_fn = lambda scores, label: tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(scores)*label, logits=scores)
        elif type in ["lsgan"]:
            self.error_fn = lambda scores, label: tf.square(scores)
        super(RelativisticGAN, self).__init__()

    def discriminator(self, real_scores, fake_scores):
        scores = real_scores - fake_scores
        loss = self.error_fn(scores, label=1.0)
        return tf.reduce_mean(loss)

    def generator(self, real_scores, fake_scores):
        scores = fake_scores - real_scores
        loss = self.error_fn(scores, label=1.0)
        return tf.reduce_mean(loss)


class RelativisticAvgGAN(object):
    def __init__(self, type="sgan"):
        self.type=type
        super(RelativisticAvgGAN, self).__init__()

    def discriminator(self, real_scores, fake_scores):
        # errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + y) ** 2))/2
        y = 1.0
        loss = tf.reduce_mean((real_scores - tf.reduce_mean(fake_scores) - y)**2) +\
               tf.reduce_mean((fake_scores - tf.reduce_mean(real_scores) + y)**2)
        return loss

    def generator(self, real_scores, fake_scores):
        #errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - y) ** 2))/2
        y = 1.0
        loss = tf.reduce_mean((real_scores - tf.reduce_mean(fake_scores) + y)**2) \
               + tf.reduce_mean((fake_scores - tf.reduce_mean(real_scores) - y)**2)
        return loss