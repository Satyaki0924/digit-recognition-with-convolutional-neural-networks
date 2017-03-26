import numpy as np
from sklearn import preprocessing


class PreProcess(object):
    def __init__(self, x):
        self.x = x
        self.x = np.array(self.x, dtype=float)

    def normalize(self):
        self.x = (self.x - self.x.min()) / (self.x.max() - self.x.min())
        return self.x

    def one_hot(self):
        one_hot = preprocessing.LabelBinarizer()
        one_hot.fit(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        return one_hot.transform(self.x)
