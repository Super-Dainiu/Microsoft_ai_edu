import numpy as np
from Neural_1_0.mat_3D import *


class Neuron(object):
    def __init__(self, param):
        self.data = None
        self.eta = param.eta
        self.insize = param.insize
        self.outsize = param.outsize
        self.batch = param.batch
        self.epsilon = param.epsilon
        self.iter = param.iter
        self.w = np.matrix(np.zeros(self.insize + 1)).T
        self.graph = None

    def reformat(self, param):
        self.eta = param.eta
        self.insize = param.insize
        self.outsize = param.outsize
        self.batch = param.batch
        self.epsilon = param.epsilon
        self.iter = param.iter
        self.w = np.matrix(np.zeros(self.insize + 1)).T

    def fetch_data(self, data):
        self.data = data
        self.graph = Mat_3D(data)

    def clear_data(self):
        self.data = None
        self.graph = None
        self.w = np.matrix(np.zeros(self.insize + 1)).T

    def __forward(self, X):
        Z = X * self.w
        return Z

    def __backward(self, X, Y, Z):
        dZ = Y-Z
        dw = (X.T*dZ)
        dw = dw/np.linalg.norm(dw, ord=2)
        return dw

    def __upgrade(self, dw, eta):
        self.w = self.w + eta*dw

    def train(self):
        datasize = len(self.data.X)
        cur = 0
        if self.batch == -1:
            for i in range(self.iter):
                self.graph.show(self.w[1, 0], self.w[2, 0], self.w[0, 0], self.data.minmax)
                batch_x = np.matrix(np.column_stack((np.ones(datasize), self.data.X)))
                batch_y = np.matrix(self.data.Y)
                batch_z = self.__forward(batch_x)
                dw = self.__backward(batch_x, batch_y, batch_z)
                self.__upgrade(dw, self.eta/np.sqrt(20*i+1))
        else:
            for i in range(self.iter):
                if i < 10 or (i % 10 == 9 and i < 100) or (i % 100 == 99 and i < 1000) or (i % 1000 == 999 and i < 10000) or (i % 10000 == 9999 and i < 100000) or i % 100000 == 99999:
                    self.graph.show(self.w[1, 0], self.w[2, 0], self.w[0, 0], self.data.minmax, i+1)
                batched = cur + self.batch
                if batched < datasize:
                    batch_x = np.column_stack((np.ones(self.batch), self.data.X[cur:batched]))
                    batch_y = np.matrix(self.data.Y[cur:batched])
                    cur = batched
                else:
                    batched = batched % datasize
                    if batched == 0:
                        batch_x = np.column_stack((np.ones(self.batch), self.data.X[cur:]))
                        batch_y = np.matrix(np.column_stack(self.data.Y[cur:])).T
                    else:
                        batch_x = np.column_stack((np.ones(self.batch), self.data.X[cur:], self.data.X[:batched]))
                        batch_y = np.matrix(np.column_stack((self.data.Y[cur:], self.data.Y[:batched]))).T
                    cur = batched
                batch_x = np.matrix(batch_x)
                batch_y = np.matrix(batch_y)
                batch_z = self.__forward(batch_x)
                dw = self.__backward(batch_x, batch_y, batch_z)
                self.__upgrade(dw, self.eta/np.sqrt(20*i+1))

    def show_result(self):
        print(self.w)

    def solution(self):
        datasize = len(self.data.X)
        X = np.matrix(np.column_stack((np.ones(datasize), self.data.X)))
        Y = np.matrix(self.data.Y)
        print(np.linalg.inv(X.T*X)*X.T*Y)