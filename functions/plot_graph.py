import os
import re

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


class PlotGraph(object):
    def __init__(self):
        pass

    @staticmethod
    def plot():
        try:

            files = os.listdir(os.path.abspath(os.curdir) + '/graphs/')
            if files is None:
                ac = 1
                l = 1
                t = 1
            else:
                ac = []
                l = []
                t = []
                for i in files:
                    if re.search('Accuracy', i):
                        ac.append(i)
                    elif re.search('Loss', i):
                        l.append(i)
                    elif re.search('Time', i):
                        t.append(i)
                ac = len(ac) + 1
                l = len(l) + 1
            loss = []
            accuracy = []
            time = []
            with open(os.path.abspath(os.curdir) + '/points/loss.txt') as f:
                loss.extend(f.read().split('\n'))
            with open(os.path.abspath(os.curdir) + '/points/accuracy.txt') as f:
                accuracy.extend(f.read().split('\n'))
            with open(os.path.abspath(os.curdir) + '/points/time.txt') as f:
                time.extend(f.read().split('\n'))
            plt.cla()
            plt.plot([float(i) for i in loss if i], c='r')
            plt.legend('Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.savefig(os.path.abspath(os.curdir) + '/graphs/Loss' + str(l) + '.png')
            print('Graph saved at: ' + os.path.abspath(os.curdir) + '/graphs/Loss' + str(l) + '.png')
            plt.close()
            plt.cla()
            plt.plot([float(i) for i in accuracy if i], c='g')
            plt.legend('Accuracy')
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.savefig(os.path.abspath(os.curdir) + '/graphs/Accuracy' + str(ac) + '.png')
            print('Graph saved at: ' + os.path.abspath(os.curdir) + '/graphs/Accuracy' + str(ac) + '.png')
            plt.close()
            plt.cla()
            plt.plot([float(i) for i in time if i], c='b')
            plt.legend('Time')
            plt.xlabel('Iterations')
            plt.ylabel('Time')
            plt.savefig(os.path.abspath(os.curdir) + '/graphs/Time' + str(ac) + '.png')
            print('Graph saved at: ' + os.path.abspath(os.curdir) + '/graphs/Time' + str(ac) + '.png')
            plt.close()
        except Exception as exception:
            print('*** Error: ' + str(exception) + ' .Run setup.sh and then train! ***')
