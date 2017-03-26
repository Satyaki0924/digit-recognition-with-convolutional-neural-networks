import tensorflow as tf

from functions.convolutionmax2d import ConvolutionMax2d

tf.reset_default_graph()


class ConvNet(object):
    def __init__(self, x, weights, biases, dropout):
        self.x = x
        self.weights = weights
        self.biases = biases
        self.dropout = dropout

    def conv_net(self):
        try:
            self.x = tf.reshape(self.x, [-1, 28, 28, 1])
            conv1 = ConvolutionMax2d(self.x, self.weights['wc1'],
                                     self.biases['bc1'], strides=1, k=2).max_pool2d()
            conv2 = ConvolutionMax2d(conv1, self.weights['wc2'],
                                     self.biases['bc2'], strides=1, k=2).max_pool2d()
            fcon1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
            fcon1 = tf.add(tf.matmul(fcon1, self.weights['wd1']), self.biases['bd1'])
            fcon1 = tf.nn.relu(fcon1)
            fcon1 = tf.nn.dropout(fcon1, self.dropout)
            out = tf.add(tf.matmul(fcon1, self.weights['out']), self.biases['out'])
            return out
        except Exception as e:
            print('*** Error: ' + str(e) + ' .Try again! ***')
            pass
