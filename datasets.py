import torch
import numpy as np
from numpy import pi
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

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

class Circles(Dataset):
    def __init__(self, n_circles=2, length=400) -> None:
        super().__init__()
        self.N = length
        self.n_circles = n_circles
        pi_t = torch.tensor(pi)
        self.theta = torch.sqrt(torch.rand(self.N))*2*pi_t # np.linspace(0,2*pi,100)

        rs = torch.linspace(1, 1 * self.n_circles, self.n_circles)
        datas = [torch.vstack([torch.cos(self.theta)*rs[i], torch.sin(self.theta)*rs[i]]).T for i in range(self.n_circles)]
        xs = [datas[i] + torch.randn(self.N,2) for i in range(self.n_circles)]
        xs_and_labels = [torch.cat((xs[i], i * torch.ones((self.N,1))), dim=1) for i in range(self.n_circles)]
        self.dataset_tensor = torch.cat(xs_and_labels, dim=0)
    
    def __len__(self):
        return self.dataset_tensor.shape[0]

    def __getitem__(self, index):
        return self.dataset_tensor[index,:2], self.dataset_tensor[index, 2].unsqueeze(0)

class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        image, label = item
        image = ToTensor()(image)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label