from cProfile import label
import matplotlib
import torch
from torch import nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
import matplotlib

@torch.no_grad()
def classifier_plot(model: nn.Module, dataset: Dataset):
    data = torch.vstack([x for x, label in dataset])
    xmax = data[:,0].amax()
    ymax = data[:,1].amax()
    xmin = data[:,0].amin()
    ymin = data[:,1].amin()
    labels = torch.vstack([label for x, label in dataset])
    set1_ind = torch.where(labels.flatten() == 1)[0]
    set0_ind = torch.where(labels.flatten() == 0)[0]

    x, y = torch.meshgrid(torch.linspace(xmin, xmax, 100), torch.linspace(ymin, ymax, 100), indexing='xy')
    model_in: torch.Tensor = torch.stack((x.flatten(), y.flatten()), dim=1)
    pred: torch.Tensor = model(model_in)
    pred = pred.reshape(100, 100)

    fig = plt.figure()
    im = plt.imshow(pred, cmap='viridis', extent=(xmin, xmax, ymax, ymin))
    ax = plt.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.scatter(data[set1_ind,0], data[set1_ind,1], color='red')
    plt.scatter(data[set0_ind,0], data[set0_ind,1], color='blue')
    plt.colorbar()
    return fig