class Param(object):
    def __init__(self, data, batch=1, eta=0.1, epsilon=0.1, epoch=100):
        self.insize = len(data.X[0])
        self.outsize = len(data.Y[0])
        self.batch = batch
        self.eta = eta
        self.epsilon = epsilon
        if batch == -1:
            self.iter = epoch
        else:
            self.iter = epoch*len(data.X)//batch