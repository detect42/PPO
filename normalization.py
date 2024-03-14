import numpy as np

class RunningMeanStd:
    def __init__(self,shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self,x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = np.zeros_like(x)
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / (self.n - 1))

class Normalization:
    def __init__(self,shape):
        self.running_ms = RunningMeanStd(shape = shape)

    def __call__(self,x,update = True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        if self.running_ms.std.all() == 0:
            return x #! 当只有一个样本时，std=0，会导致除0错误，此时无所谓normalization，因此直接返回x
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class RewardScaling:
    def __init__(self,shape,gamma):
        self.shape = shape # reward shape is 1
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape = self.shape)
        self.R = np.zeros(self.shape)
    def __call__(self,x):
        self.R = self.gamma * self.R +x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8) # Only divided std
        return x
    def reset(self):
        self.R = np.zeros(self.shape)