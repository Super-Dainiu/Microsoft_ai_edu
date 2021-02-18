from Neural_2_0.neuron import *
from Neural_2_0.csv_reader import *
from Neural_2_0.hyperparam import *
import os


class Instructions:
    def __init__(self):
        self.exit = False
        self.instructions = {"Set universal parameter": self.set_parameter,
                             "Set default data": self.input_data,
                             "Create neuron": self.create_neuron,
                             "Train neuron": self.train_neuron,
                             "Use neuron": self.use_neuron,
                             "Reformat neuron": self.reformat_neuron,
                             "Choose neurons": self.choose_neuron,
                             "Show train history": self.plot_train_history,
                             "Destroy neuron": self.destroy_neuron,
                             "Exit": exit}
        self.data = CSVReader(os.getcwd() + "\\data\\iris.csv")
        self.param = Hyperparam(self.data)
        self.neuron = {}
        self.cur = None
        self.curname = "Null"
        self.count = 0
        print("Welcome to Neuron_2_0")

    def use_neuron(self):
        if self.cur:
            x = (float(input("%s = " % label)) for label in self.cur.data.head)
            print(self.cur.classify(x))

    def destroy_neuron(self):
        if self.neuron:
            for name in self.neuron.keys():
                print(name, self.neuron[name].state)
            choice = input("Choose neuron: ")
            if choice in self.neuron.keys():
                self.neuron.pop(choice)
            else:
                print("Error! No such neuron!")
            if choice == self.curname:
                self.curname = "Null"
                self.cur = None
        else:
            print("\tNull")

    def reformat_neuron(self):
        if self.cur:
            self.cur.reformat(Hyperparam(self.cur.data, batch=int(input("batch = ")), eta=float(input("eta = ")),
                                         epoch=int(input("epoch = "))), self.cur.data)

    def create_neuron(self):
        if input("Use your default dataset (Y/N)? ") in ["y", "Y", '']:
            data = self.data
            param = self.param
        else:
            data = CSVReader(os.getcwd() + "\\data\\" + input("Your dataset: "))
            param = Hyperparam(data, batch=int(input("batch = ")), eta=float(input("eta = ")),
                               epoch=int(input("epoch = ")))
        name = input("Name: ")
        if name:
            self.curname = name
        else:
            self.count += 1
            self.curname = "n_%d" % self.count
        self.cur = Neuron(param, data)
        self.neuron[self.curname] = self.cur

    def train_neuron(self):
        if self.cur:
            self.cur.train()

    def choose_neuron(self):
        if self.neuron:
            for name in self.neuron.keys():
                print(name, self.neuron[name].state)
            choice = input("Choose neuron: ")
            if choice in self.neuron.keys():
                self.cur = self.neuron[choice]
                self.curname = choice
            else:
                print("Error! No such neuron!")
        else:
            print("\tNull")

    def get_instruction(self):
        print("Instruction:")
        for (i, instruction) in enumerate(self.instructions.keys()):
            print("\t%d.%s" % (i + 1, instruction))
        print("Current neuron: %s" % self.curname, end='')
        if self.cur:
            print(" (%s)" % self.cur.state, end=' ')
            self.cur.p.show_param()
        instruction = input("\nYour instruction here: ")
        if instruction in self.instructions.keys():
            self.instructions[instruction]()
        elif instruction in (str(i) for i in range(1, len(self.instructions.keys()) + 1)):
            self.instructions[list(self.instructions.keys())[int(instruction) - 1]]()
        else:
            print("Unknown instruction!")

    def set_parameter(self):
        if self.data:
            self.param = Hyperparam(self.data, batch=int(input("batch = ")), eta=float(input("eta = ")),
                                    epoch=int(input("epoch = ")))
        else:
            print("Input your data first!")
            self.input_data()
            self.set_parameter()

    def input_data(self):
        self.data = CSVReader(os.getcwd() + "\\data\\" + input("Your dataset: "))
        self.param = Hyperparam(self.data)

    def plot_train_history(self):
        if self.cur and self.cur.state == "trained":
            self.cur.show_history()
