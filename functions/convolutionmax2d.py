import tensorflow as tf

tf.reset_default_graph()


class ConvolutionMax2d(object):
    def __init__(self, x, W, b, strides, k):
        self.x = x
        self.W = W
        self.b = b
        self.strides = strides
        self.k = k

    def conv2d(self):
        x = tf.nn.conv2d(self.x, self.W, strides=[1, self.strides, self.strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.b)
        self.x = tf.nn.relu(x)

    def max_pool2d(self):
        self.conv2d()
        return tf.nn.max_pool(self.x, ksize=[1, self.k, self.k, 1],
                              strides=[1, self.k, self.k, 1], padding='SAME')
