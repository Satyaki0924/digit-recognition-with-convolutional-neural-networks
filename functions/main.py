import os
import sys

import tensorflow as tf

from functions.plot_graph import PlotGraph
from functions.run_net import RunNet


class Main(object):
    def __init__(self, learning_rate=0.001,
                 training_iters=200000,
                 batch_size=128,
                 display_step=10,
                 n_input=784,
                 n_classes=10,
                 dropout=0.75):
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.n_input = n_input
        self.n_classes = n_classes
        self.dropout = dropout
        self.checkpoint = None
        self.program = None
        self.path = os.path.dirname(os.path.abspath(__file__))

    def get_values(self):
        """
        Set the following parameters
        :return: None
        """
        while True:
            print('*****Author: Satyaki Sanyal*****')
            print('***This project must only be used for educational purpose***')
            print('\r>> Enter 1: to Train, '
                  '2: to Test, '
                  '3: to generate Graph, '
                  '4: to BREAK')
            arr = [1, 2, 3, 4]
            try:
                i = int(input('>> '))
                if i in arr:
                    print(i)
                    self.program = i
                    if i == 2:
                        check = tf.train.get_checkpoint_state(self.path + '/../check_points/')
                        if check:
                            print(check)
                            print('>> Enter the checkpoint model number (ex: 195840)')
                            while True:
                                try:
                                    c = int(input('>> '))
                                    if tf.train.checkpoint_exists(self.path + '/../check_points/model_' + str(
                                            c) + '.ckpt'):
                                        sys.stdout.write('\rCheckpoint accepted...')
                                        sys.stdout.write('\rTesting output...')
                                        print('\n')
                                        self.checkpoint = str(c)
                                        break
                                    else:
                                        print('*** Error: Model not found. Re-enter ***')
                                except Exception as exception1:
                                    print('*** Error: ' + str(exception1) + ' .Try again! ***')
                                    pass
                        else:
                            print('*** Error: Please train the model. No checkpoints found! ***')
                    break
                else:
                    print("*** Error: Didn't recognise your input ***")

            except Exception as e:
                print('*** Error: ' + str(e) + ' .Try again! ***')
                pass

    def main(self):
        try:
            while True:
                self.get_values()
                if self.program == 4:
                    break
                if self.program == 3:
                    PlotGraph().plot()
                    break
                arr = [1, 2]
                if self.program in arr:
                    if self.program == 1:
                        self.program = 'train'
                    elif self.program == 2:
                        self.program = 'test'
                if not self.checkpoint:
                    self.checkpoint = 1
                RunNet(self.learning_rate, self.training_iters, self.batch_size,
                       self.display_step, self.n_input, self.n_classes,
                       self.dropout, self.program, self.checkpoint).run()
                if self.program == 'train':
                    break
        except Exception as e:
            print('***Error: ' + str(e) + '***')