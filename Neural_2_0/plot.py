import numpy as np
import matplotlib.pyplot as plt


class Graph(object):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = None

    def loss_function_history(self, history):
        plt.clf()
        self.ax = self.fig.gca(fc='whitesmoke', projection='3d')
        x = [i for (i, _, _) in history]
        y = [i for (_, i, _) in history]
        z = [i for (_, _, i) in history]
        self.ax.scatter(x, y, z)
        plt.title("Loss function history")
        plt.xlabel("Layer_1 descent")
        plt.ylabel("Layer_2 descent")
        plt.show()