import numpy as np
import pandas as pd
from pathlib import Path


class CSVReader(object):
    def __init__(self, file):
        self.file = Path(file)
        self.X = None
        self.Y = None
        self.cat = []
        self.head = []
        self.length = 0
        self.order = []
        if self.file.exists():
            data = pd.read_csv(self.file, delimiter=',')
            self.order = data.pop("class")
            self.cat = list(set(self.order))
            self.head = list(data.head().keys())
            self.length = len(self.order)
            self.X = np.column_stack([data[head] for head in self.head])
            self.Y = np.column_stack([np.where(self.order == name, 1, 0) for name in self.cat])
            self.shuffle()
        else:
            raise Exception("Cannot find train file!!!")

    def shuffle(self):
        seed = np.random.randint(1, 114514)
        np.random.seed(seed)
        self.X = np.random.permutation(self.X)
        np.random.seed(seed)
        self.Y = np.random.permutation(self.Y)