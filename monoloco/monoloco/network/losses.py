
import math
import torch
import numpy as np
import matplotlib.pyplot as plt


class CustomL1Loss(torch.nn.Module):
    """
    L1 loss with more weight to errors at a shorter distance
    It inherits from nn.module so it supports backward
    """

    def __init__(self, dic_norm, device, beta=1):
        super(CustomL1Loss, self).__init__()

        self.dic_norm = dic_norm
        self.device = device
        self.beta = beta

    @staticmethod
    def compute_weights(xx, beta=1):
        """
        Return the appropriate weight depending on the distance and the hyperparameter chosen
        alpha = 1 refers to the curve of A Photogrammetric Approach for Real-time...
        It is made for unnormalized outputs (to be more understandable)
        From 70 meters on every value is weighted the same (0.1**beta)
        Alpha is optional value from Focal loss. Yet to be analyzed
        """
        # alpha = np.maximum(1, 10 ** (beta - 1))
        alpha = 1
        ww = np.maximum(0.1, 1 - xx / 78)**beta

        return alpha * ww

    def print_loss(self):
        xx = np.linspace(0, 80, 100)
        y1 = self.compute_weights(xx, beta=1)
        y2 = self.compute_weights(xx, beta=2)
        y3 = self.compute_weights(xx, beta=3)
        plt.plot(xx, y1)
        plt.plot(xx, y2)
        plt.plot(xx, y3)
        plt.xlabel("Distance [m]")
        plt.ylabel("Loss function Weight")
        plt.legend(("Beta = 1", "Beta = 2", "Beta = 3"))
        plt.show()

    def forward(self, output, target):

        unnormalized_output = output.cpu().detach().numpy() * self.dic_norm['std']['Y'] + self.dic_norm['mean']['Y']
        weights_np = self.compute_weights(unnormalized_output, self.beta)
        weights = torch.from_numpy(weights_np).float().to(self.device)  # To make weights in the same cuda device
        losses = torch.abs(output - target) * weights
        loss = losses.mean()  # Mean over the batch
        return loss


class LaplacianLoss(torch.nn.Module):
    """1D Gaussian with std depending on the absolute distance
    """
    def __init__(self, size_average=True, reduce=True, evaluate=False):
        super(LaplacianLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.evaluate = evaluate

    def laplacian_1d(self, mu_si, xx):
        """
        1D Gaussian Loss. f(x | mu, sigma). The network outputs mu and sigma. X is the ground truth distance.
        This supports backward().
        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py

        """
        mu, si = mu_si[:, 0:1], mu_si[:, 1:2]
        # norm = xx - mu
        norm = 1 - mu / xx  # Relative

        term_a = torch.abs(norm) * torch.exp(-si)
        term_b = si
        norm_bi = (np.mean(np.abs(norm.cpu().detach().numpy())), np.mean(torch.exp(si).cpu().detach().numpy()))

        if self.evaluate:
            return norm_bi
        return term_a + term_b

    def forward(self, outputs, targets):

        values = self.laplacian_1d(outputs, targets)

        if not self.reduce or self.evaluate:
            return values
        if self.size_average:
            mean_values = torch.mean(values)
            return mean_values
        return torch.sum(values)


class GaussianLoss(torch.nn.Module):
    """1D Gaussian with std depending on the absolute distance
    """
    def __init__(self, device, size_average=True, reduce=True, evaluate=False):
        super(GaussianLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.evaluate = evaluate
        self.device = device

    def gaussian_1d(self, mu_si, xx):
        """
        1D Gaussian Loss. f(x | mu, sigma). The network outputs mu and sigma. X is the ground truth distance.
        This supports backward().
        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py
        """
        mu, si = mu_si[:, 0:1], mu_si[:, 1:2]

        min_si = torch.ones(si.size()).cuda(self.device) * 0.1
        si = torch.max(min_si, si)
        norm = xx - mu
        term_a = (norm / si)**2 / 2
        term_b = torch.log(si * math.sqrt(2 * math.pi))

        norm_si = (np.mean(np.abs(norm.cpu().detach().numpy())), np.mean(si.cpu().detach().numpy()))

        if self.evaluate:
            return norm_si

        return term_a + term_b

    def forward(self, outputs, targets):

        values = self.gaussian_1d(outputs, targets)

        if not self.reduce or self.evaluate:
            return values
        if self.size_average:
            mean_values = torch.mean(values)
            return mean_values
        return torch.sum(values)
