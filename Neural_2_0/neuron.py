import numpy as np
from Neural_2_0.plot import *


class Sigmoid(object):
    def forward(self, Z):
        A = 1.0 / (1.0 + np.exp(-Z))
        return A

    def backward(self, A):
        dA = np.multiply(A, 1 - A)
        return dA


class Softmax(object):
    def forward(self, Z):
        e_z = np.exp(Z)
        A = np.exp(Z) / np.sum(e_z, axis=1)
        return A


class Neuron(object):
    def __init__(self, param, data):
        self.p = param
        self.data = data
        self.W1 = np.matrix(np.zeros((self.p.insize + 1, self.p.catsize)))
        self.W2 = np.matrix(np.zeros((self.p.catsize + 1, self.p.catsize)))
        self.A1 = None
        self.Z1 = None
        self.A2 = None
        self.Z2 = None
        self.layer1 = 0
        self.layer2 = 0
        self.state = "untrained"
        self.history = []

    def reformat(self, param, data):
        self.p = param
        self.data = data
        self.W1 = np.matrix(np.zeros((self.p.insize + 1) * self.p.catsize)).reshape(self.p.insize + 1, self.p.catsize)
        self.W2 = np.matrix(np.zeros((self.p.catsize + 1) * self.p.catsize)).reshape(self.p.catsize + 1, self.p.catsize)
        self.A1 = None
        self.Z1 = None
        self.A2 = None
        self.Z2 = None
        self.layer1 = 0
        self.layer2 = 0
        self.history = []
        self.state = "untrained"

    def __classify(self, x):
        Z1 = np.matrix(np.column_stack([1] + list(x))) @ self.W1
        A1 = Sigmoid().forward(Z1)
        Z2 = np.matrix(np.column_stack((1, A1))) @ self.W2
        A2 = Softmax().forward(Z2)
        return A2

    def classify(self, x):
        A2 = self.__classify(x).tolist()[0]
        cat = self.data.cat[A2.index(max(A2))]
        print(cat)

    def classific(self, x):
        A2 = self.__classify(x).tolist()[0]
        cat = self.data.cat[A2.index(max(A2))]


    def __loss_function(self):
        L = 0
        for i in range(self.p.datasize):
            x = self.data.X[i, :]
            y = self.data.Y[i, :]
            a = self.__classify(x)
            L -= y @ np.log(a).T
        return L

    def __forward(self, x, batchsize):
        self.Z1 = x @ self.W1
        self.A1 = Sigmoid().forward(self.Z1)
        self.Z2 = np.matrix(np.column_stack((np.ones(batchsize), self.A1))) @ self.W2
        self.A2 = Softmax().forward(self.Z2)

    def __backward(self, x, y, batchsize):
        dZ2 = self.A2 - y
        dW2 = np.matrix(np.column_stack((np.ones(batchsize), self.A1))).T @ dZ2
        dA1 = Sigmoid().backward(self.A1)
        dZ1 = np.multiply(dZ2 @ self.W2[1:, :].T, dA1)
        dW1 = x.T @ dZ1
        if np.linalg.norm(dW1):
            dW1 = dW1 / np.linalg.norm(dW1)
        if np.linalg.norm(dW2):
            dW2 = dW2 / np.linalg.norm(dW2)
        return dW1, dW2

    def __update(self, dW1, dW2, i, layer1=False, layer2=False):
        if layer1:
            self.W1 = self.W1 - self.p.eta * dW1 / np.sqrt(i + 1)
        if layer2:
            self.W2 = self.W2 - self.p.eta * dW2 / np.sqrt(i + 1)

    def __batch(self):
        epoch = []
        batch = self.p.batch
        if batch != -1:
            cur = 0
            while self.p.datasize - cur >= batch:
                epoch.append((np.matrix(np.column_stack((np.ones(batch), self.data.X[cur:cur + batch, :]))),
                              np.matrix(self.data.Y[cur:cur + batch, :]), self.p.batch))
                cur = cur + batch
            if cur < self.p.datasize:
                epoch.append(
                    (np.matrix(np.column_stack((np.ones(self.p.datasize - cur), self.data.X[cur:self.p.datasize, :]))),
                     np.matrix(self.data.Y[cur:self.p.datasize, :]), self.p.datasize - cur))
        else:
            epoch.append((np.matrix(np.column_stack((np.ones(self.p.datasize), self.data.X))), np.matrix(self.data.Y),
                          self.p.datasize))
        return epoch

    def train(self):
        for i in range(self.p.epoch):
            self.data.shuffle()
            batch = self.__batch()
            layer1 = np.random.choice(a=(1, 0), p=[0.6, 0.4])
            layer2 = np.random.choice(a=(1, 0), p=[0.6, 0.4])
            for (batch_x, batch_y, batchsize) in batch:
                self.__forward(batch_x, batchsize)
                dW1, dW2 = self.__backward(batch_x, batch_y, batchsize)
                self.__update(dW1, dW2, i, layer1, layer2)
            self.layer1 += layer1
            self.layer2 += layer2
            self.history.append((self.layer1, self.layer2, self.__loss_function()))
        self.state = "trained"
        self.show_history()

    def show_history(self):
        Graph().loss_function_history(self.history)
        print("Current L = %f" % self.__loss_function())