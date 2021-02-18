import numpy as np
import csv
from Neural_1_0.mat_3D import *


class CSVReader(object):
    def __init__(self, file):
        self.file = open(file)
        self.X = None
        self.Y = None
        self.minmax = []
        self.normalized = False
        if self.file:
            reader = csv.reader(self.file, delimiter=',')
            rows = [row for row in reader]
            self.title = rows.pop(0)
            X = [list(map(float, row[:-1])) for row in rows]
            Y = []
            for row in rows:
                Y.append([float(row[-1])])
            self.X = np.array(X)
            self.Y = np.array(Y)
            self.Xsize = len(self.X[0])
            for i in range(self.Xsize):
                xmax = max(self.X[:, i])
                xmin = min(self.X[:, i])
                self.minmax.append((xmax, xmin))
        else:
            raise Exception("Cannot find train file!!!")

    def normalize(self, normx = True, normy = True):
        self.minmax = []
        if normx:
            for i in range(self.Xsize):
                xmax = max(self.X[:, i])
                xmin = min(self.X[:, i])
                self.X[:, i] = (self.X[:, i]-xmin)/(xmax-xmin)
                self.minmax.append([xmax, xmin])
                self.normalized = True
        if normy:
            ymax = max(self.Y)
            ymin = min(self.Y)
            self.Y = (self.Y-ymin)/(ymax-ymin)