from __future__ import print_function

import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from functions.conv_net import ConvNet
from functions.net_init import NetInit
from functions.preprocess import PreProcess

tf.reset_default_graph()


class RunNet(object):
    def __init__(self, learning_rate, training_iters,
                 batch_size, display_step, n_input,
                 n_classes, dropout, program, check):
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.n_input = n_input
        self.n_classes = n_classes
        self.dropout = dropout
        self.program = program
        self.check = check
        init = NetInit(self.n_input, self.n_classes)
        self.x = init.neural_net_image_shape()
        self.y = init.neural_net_label_shape()
        self.keep_prob = init.neural_net_keep_prob()
        self.mnist = input_data.read_data_sets('./dataset/data/', one_hot=True)
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
            'out': tf.Variable(tf.random_normal([1024, self.n_classes]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

    def run(self):
        try:
            graph_points_loss = []
            graph_points_accuracy = []
            graph_points_time = []
            pred = ConvNet(self.x, self.weights, self.biases, self.keep_prob).conv_net()
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
            optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            saver = tf.train.Saver()
            save_file = os.path.dirname(os.path.abspath(__file__)) + '/../check_points/model'
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                if self.program == 'train':
                    try:
                        step = 1
                        start = time.time()
                        while step * self.batch_size < self.training_iters:
                            batch_x, batch_y = self.mnist.train.next_batch(self.batch_size)
                            batch_x = PreProcess(batch_x).normalize()
                            sess.run(optimiser, feed_dict={self.x: batch_x, self.y: batch_y,
                                                           self.keep_prob: self.dropout})
                            if step % self.display_step == 0:
                                loss, acc = sess.run([cost, accuracy], feed_dict={self.x: batch_x,
                                                                                  self.y: batch_y,
                                                                                  self.keep_prob: 1.})
                                current = float(time.time() - start)
                                if graph_points_time is None:
                                    graph_points_time.append(current)
                                else:
                                    graph_points_time.append(current - float(sum(graph_points_time)))

                                current_v = 'sec'
                                time_left = float(
                                    (current / float(step * self.batch_size)) * self.training_iters) - current
                                if current > 60:
                                    current /= 60
                                    current_v = 'min'
                                time_left_v = 'sec'
                                if time_left > 60:
                                    time_left /= 60
                                    time_left_v = 'min'
                                per = (float(step * self.batch_size) / float(self.training_iters))
                                sys.stdout.write("\rIter " + str(step * self.batch_size) +
                                                 ", Minibatch Loss : " + "{:.6f}".format(loss) +
                                                 ", Training Accuracy : " + "{:.4f}%".format(acc * 100) +
                                                 ", Time taken : {:.4f} ".format(current) + current_v +
                                                 ", % complete : {:.1f}%".format(per * 100) +
                                                 ", Time left : {:.4f} ".format(time_left) + time_left_v)
                                saver.save(sess, save_file + '_' + str(step * self.batch_size) + '.ckpt')
                                graph_points_loss.append(loss)
                                graph_points_accuracy.append(acc)
                            step += 1
                        saver.save(sess, save_file + '.ckpt')
                        sys.stdout.write("\rOptimization Finished! Accuracy: {:.4f}%".format(acc * 100))
                        print('\n\n')
                        with open(os.path.dirname(os.path.abspath(__file__)) + '/../points/loss.txt', 'w') as file:
                            for i in graph_points_loss:
                                file.write(str(i) + '\n')
                            file.close()
                            print('Training loss points saved')
                        with open(os.path.dirname(os.path.abspath(__file__)) + '/../points/time.txt', 'w') as file:
                            for i in graph_points_time:
                                file.write(str(i) + '\n')
                            file.close()
                            print('Training time points saved')
                        with open(os.path.dirname(os.path.abspath(__file__)) + '/../points/accuracy.txt', 'w') as file:
                            for i in graph_points_accuracy:
                                file.write(str(i) + '\n')
                            file.close()
                            print('Training accuracy points saved')
                    except Exception as e:
                        print('*** Error: ' + str(e) + ' .Try again! ***')
                        pass
                elif self.program == 'test':
                    try:
                        tf.reset_default_graph()
                        while True:
                            print('>> Enter test size (max-size: {:.1f})'.format(len(self.mnist.test.images)))
                            size = int(input('>> '))
                            if size <= 10000:
                                break
                        saver.restore(sess, save_file + '_' + self.check + '.ckpt')
                        accrcy = sess.run(accuracy, feed_dict={self.x: self.mnist.test.images[:size],
                                                               self.y: self.mnist.test.labels[:size],
                                                               self.keep_prob: 1.})
                        print("Testing Accuracy: {:.4f}%".format(accrcy * 100))
                    except Exception as e:
                        print('*** Error: ' + str(e) + ' .Try again! ***')
                        pass
                else:
                    print('*** Error: Unknown program id. Cannot execute!***')
        except:
            pass
