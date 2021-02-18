import matplotlib.pyplot as plt
import numpy as np


class Mat_3D:
    def __init__(self, data):
        self.X = data.X
        self.Y = data.Y
        self.fig = plt.figure()
        self.ax = self.fig.gca(fc='whitesmoke', projection='3d')

    def show(self, w1, w2, b, minmax, i):
        plt.cla()
        self.ax.scatter(self.X[:, 0], self.X[:, 1], self.Y)
        x1 = np.linspace(*minmax[0], 150)
        x2 = np.linspace(*minmax[1], 150)
        X, Y = np.meshgrid(x1, x2)
        self.ax.plot_surface(X, Y, Z=X*w1+Y*w2+b, color="g", alpha=0.6)
        self.ax.set_zlim(-400, 400)
        plt.title('%d iter' %i)
        plt.pause(0.01)