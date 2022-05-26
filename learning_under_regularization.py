import os
import torch
import hydra
import torch.nn as nn
from torch.optim import SGD
import pytorch_lightning as pl
from torch.nn import functional as F
import torchmetrics
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

class LipschitzRegularizer(nn.Module):
    def __init__(self, l_const=1):
        super().__init__()
        self.l_const = torch.tensor(l_const, dtype=torch.float)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs.norm(p=2, dim=1, keepdim=False)

class RegularizedMnistClassifier(pl.LightningModule):
    def __init__(self, classifier):
        super().__init__()
        self.classifier: nn.Module = classifier
        self.lr = 0.001
        self.regularizer = None
        self.train_accuracy = torchmetrics.Accuracy(num_classes=10)
        self.val_accuracy = torchmetrics.Accuracy(num_classes=10)

    def forward(self, inputs):
        return self.classifier(inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs)
        loss = F.cross_entropy(preds, F.one_hot(labels, preds.shape[1]).float())
        loss = loss if self.regularizer is None else loss + self.regularizer
        self.train_accuracy(preds.argmax(dim=1),labels)
        self.log("loss", loss)
        self.log("accuracy", self.train_accuracy, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds: torch.Tensor = self(inputs)
        loss = F.cross_entropy(preds, F.one_hot(labels, preds.shape[1]).float())
        self.val_accuracy(preds.argmax(dim=1),labels)
        self.log("val loss", loss, prog_bar=True)
        self.log("val accuracy", self.val_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        return SGD(self.classifier.parameters(), self.lr)

@hydra.main(config_path="configs", config_name="config")
def myapp(cfg: DictConfig):
    transform = ToTensor()
    train_dataset = MNIST(os.path.expanduser("~/datasets/MNIST"), train=True, download=True, transform=transform)
    val_dataset = MNIST(os.path.expanduser("~/datasets/MNIST"), train=False, download=True, transform=transform)
    classifier = nn.Sequential(
                            nn.Conv2d(1, cfg.channels, (3,3)),
                            nn.Flatten(),
                            nn.LazyLinear(10))
    model = RegularizedMnistClassifier(classifier)
    trainer = pl.Trainer(logger=TensorBoardLogger("tb_logs"), enable_progress_bar=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=8)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    
    myapp()