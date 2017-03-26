import tensorflow as tf


class NetInit(object):
    def __init__(self, x, n_classes):
        self.x = x
        self.n_classes = n_classes

    def neural_net_image_shape(self):
        return tf.placeholder(tf.float32, shape=[None, self.x], name='x')

    def neural_net_label_shape(self):
        return tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y')

    @staticmethod
    def neural_net_keep_prob():
        return tf.placeholder(tf.float32, shape=None, name='keep_prob')
