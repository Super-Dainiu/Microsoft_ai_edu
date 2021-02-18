class Hyperparam(object):
    def __init__(self, data, batch=15, eta=0.4, epoch=600):
        self.insize = len(data.head)
        self.catsize = len(data.cat)
        self.batch = batch
        self.eta = eta
        self.epoch = epoch
        self.datasize = data.length

    def show_param(self):
        print("\b\tbatch = %d\teta = %f\tepoch = %d\tdatasize = %d" % (self.batch, self.eta, self.epoch, self.datasize), end='')
