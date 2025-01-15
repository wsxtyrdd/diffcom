import torch


class MSE(torch.nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.squared_difference = torch.nn.MSELoss(reduction='none')

    def forward(self, X, Y):
        return torch.mean(self.squared_difference(X * 255., Y * 255.))  # / 255.


class Distortion(torch.nn.Module):
    def __init__(self, config):
        super(Distortion, self).__init__()
        if config.distortion_metric == 'MSE':
            self.dist = MSE()
        else:
            print("Unknown distortion type!")
            raise ValueError

    def forward(self, X, Y):
        return self.dist.forward(X, Y)  # / 255.
