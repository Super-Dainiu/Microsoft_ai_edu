from Neural_1_0.format import *
from Neural_1_0.csv_reader import *
from Neural_1_0.neuron import *
from Neural_1_0.mat_3D import *
import os

if __name__ == "__main__":
    data = CSVReader(os.getcwd()+"\\data\\mlm.csv")
    param = Param(data, batch=5, epoch=5000, eta=10)
    neuron = Neuron(param)
    neuron.fetch_data(data)
    neuron.train()
    neuron.show_result()
    neuron.solution()
    plt.show()