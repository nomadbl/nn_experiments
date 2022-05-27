import torch
import numpy as np
from numpy import pi
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class Spiral(Dataset):
    def __init__(self, length=400) -> None:
        super().__init__()
        self.N = length
        pi_t = torch.tensor(pi)
        self.theta = torch.sqrt(torch.rand(self.N))*2*pi_t # np.linspace(0,2*pi,100)

        r_a = 2*self.theta + pi_t
        data_a = torch.vstack([torch.cos(self.theta)*r_a, torch.sin(self.theta)*r_a]).T
        x_a = data_a + torch.randn(self.N,2)

        r_b = -2*self.theta - pi_t
        data_b = torch.vstack([torch.cos(self.theta)*r_b, torch.sin(self.theta)*r_b]).T
        x_b = data_b + torch.randn(self.N,2)

        res_a = torch.cat((x_a, torch.zeros((self.N,1))), dim=1)
        res_b = torch.cat((x_b, torch.ones((self.N,1))), dim=1)

        self.res = torch.cat([res_a, res_b], dim=0)
    
    def __len__(self):
        return self.res.shape[0]

    def __getitem__(self, index):
        return self.res[index,:2], self.res[index, 2].unsqueeze(0)

# plt.scatter(x_a[:,0],x_a[:,1])
# plt.scatter(x_b[:,0],x_b[:,1])
# plt.show()