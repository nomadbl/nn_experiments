import os

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

from self_normalizing_nn import self_normalizing_nn_init

from corrected_adam import CorrectedAdam
from SAG_optimizer import SAG

class MnistClassifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.accuracy = Accuracy(num_classes=10)
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = 0.001

        def block(in_dim, out_dim, stride=1):
            conv = torch.nn.Conv2d(in_dim, out_dim, 3, stride, 1)
            torch.nn.init.xavier_uniform_(conv.weight, gain=torch.nn.init.calculate_gain("relu"))
            torch.nn.init.zeros_(conv.bias)
            # self_normalizing_nn_init(conv)
            return torch.nn.Sequential(conv, 
                                       torch.nn.BatchNorm2d(out_dim), 
                                       torch.nn.ReLU())

        # for digits mnist
        layers = [(1, 16, 1), (16, 16, 1), (16, 32, 2), (32, 32, 2), (32, 64, 2)]
        middle = torch.nn.Sequential(*[block(_id, _od, s) for _id, _od, s in layers])
        end = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1,1)),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(64, 10))
        torch.nn.init.xavier_uniform_(end[2].weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.zeros_(end[2].bias)
        # self_normalizing_nn_init(end[2])
        

        # def fblock(in_dim, out_dim):
        #     conv = torch.nn.Conv2d(in_dim, out_dim, 5, 1, 0)
        #     # torch.nn.init.xavier_normal_(conv.weight, gain=torch.nn.init.calculate_gain("relu"))
        #     # torch.nn.init.zeros_(conv.bias)
        #     self_normalizing_nn_init(conv)
        #     return torch.nn.Sequential(conv,
        #                                torch.nn.SELU(),
        #                                torch.nn.MaxPool2d((3, 3), 2),
        #                             #    torch.nn.BatchNorm2d(out_dim), 
        #                                )

        # # for fashion mnist
        # # in 28 x 28
        # layers = [(1, 32), # out 12 x 12
        #           (32, 64)] # out 4 x 4
        # middle = torch.nn.Sequential(*[fblock(_id, _od) for _id, _od in layers])
        # end = torch.nn.Sequential(torch.nn.Flatten(),
        #                           torch.nn.Linear(576, 1024), 
        #                           torch.nn.SELU(),
        #                           torch.nn.Linear(1024, 10))
        # # torch.nn.init.xavier_normal_(end[1].weight, gain=torch.nn.init.calculate_gain("relu"))
        # # torch.nn.init.zeros_(end[1].bias)
        # self_normalizing_nn_init(end[1])
        # # torch.nn.init.xavier_normal_(end[3].weight, gain=torch.nn.init.calculate_gain("relu"))
        # # torch.nn.init.zeros_(end[3].bias)
        # self_normalizing_nn_init(end[3])

        self.model = torch.nn.Sequential(middle, end)

    def configure_optimizers(self):
        # return CorrectedAdam(self.model.parameters(), self.lr, (0.9, 0.3), eps=0.1)
        return SAG(self.model.parameters(), 64, self.lr, (0.9, 0.9), tau=3)

    def forward(self, inputs):
        return self.model(inputs)
    
    def on_after_backward(self):
        # print("on_after_backward")
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}.grad = {param.grad.norm()}")
        ...
    
    def training_step(self, batch, batch_idx):
        x, label = batch
        y = self(x)
        loss = F.cross_entropy(y, label)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, label = batch
        y = self(x)
        loss = F.cross_entropy(y, label)
        self.log("val/loss", loss)
        self.accuracy(y, label)
        self.log("val/accuracy", self.accuracy.compute())


def get_dataloaders(fashion=False):
    if fashion:
        transform = Compose([ToTensor(), Normalize((0), (256))])
    else:
        transform = Compose([Resize((32, 32)), ToTensor(), Normalize((0), (256))])
    ds = FashionMNIST if fashion else MNIST
    dataset_location = "Fashion_MNIST_data" if fashion else "MNIST_data"
    train_dataset = ds(root=os.path.join(os.getcwd(), dataset_location), transform=transform, train=True, download=True)
    val_dataset = ds(root=os.path.join(os.getcwd(), dataset_location), transform=transform, train=False, download=True)
    train_loader = DataLoader(train_dataset, 64, True, num_workers=2)
    val_loader = DataLoader(val_dataset, 64, False, num_workers=2)
    return train_loader, val_loader

def train(fashion):
    train_loader, val_loader = get_dataloaders(fashion=fashion)
    model = MnistClassifier()
    if fashion:
        # trainer = pl.Trainer(logger=TensorBoardLogger("fashion_mnist_corrected_adam"),
        #                  max_epochs=100)
        trainer = pl.Trainer(logger=TensorBoardLogger("fashion_mnist_sag"),
                         max_epochs=100)
    else:
        # trainer = pl.Trainer(logger=TensorBoardLogger("mnist_corrected_adam"),
        #                      max_epochs=100)
        trainer = pl.Trainer(logger=TensorBoardLogger("mnist_sag"),
                             max_epochs=100)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    fashion=False
    train(fashion)