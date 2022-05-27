import imp
import os
import torch
import hydra
import torchmetrics
import torch.nn as nn
from torch.optim import SGD
import albumentations as aug
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.nn import functional as F
from albumentations.pytorch import ToTensorV2

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning.loggers import TensorBoardLogger

from datasets import Spiral
from plots import classifier_plot

class LipschitzRegularizer(nn.Module):
    def __init__(self, l_const=1):
        super().__init__()
        self.l_const = torch.tensor(l_const, dtype=torch.float)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        flat_inputs = torch.flatten(inputs, 1)
        flat_targets = torch.flatten(targets, 1)
        permutation = torch.randperm(inputs.shape[0])
        input_differences = flat_inputs - flat_inputs.index_select(0, permutation)
        target_differences = flat_targets - flat_targets.index_select(0, permutation)
        input_norm = input_differences.norm(p=2, dim=1, keepdim=False)
        target_norm = target_differences.norm(p=2, dim=1, keepdim=False)
        
        loss: torch.Tensor = target_norm - input_norm * self.l_const
        # target_norm - input_norm * self.l_const - 1 < 0
        # return lhs as loss if > 0
        loss = loss.maximum(torch.zeros_like(loss)).mean()
        return loss

class RegularizedMnistClassifier(pl.LightningModule):
    def __init__(self, classifier, cfg):
        super().__init__()
        self.classifier: nn.Module = classifier
        self.lr = 0.0001
        self.regularizer = LipschitzRegularizer()
        self.train_accuracy = torchmetrics.Accuracy(num_classes=1)
        self.val_accuracy = torchmetrics.Accuracy(num_classes=1)
        self.reg_coeff = cfg.reg_coeff

    def forward(self, inputs):
        return self.classifier(inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs)
        loss = F.binary_cross_entropy_with_logits(preds, labels)
        reg = 0 if self.regularizer is None else self.regularizer(inputs, preds)
        loss = loss if self.regularizer is None else loss + reg * self.reg_coeff
        self.train_accuracy(preds.flatten(),labels.int().flatten())
        self.log("loss", loss)
        self.log("regularization loss", reg, prog_bar=True)
        self.log("accuracy", self.train_accuracy, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds: torch.Tensor = self(inputs)
        loss = F.binary_cross_entropy_with_logits(preds, labels)
        self.val_accuracy(preds.flatten(),labels.int().flatten())
        self.log("val loss", loss, prog_bar=True)
        self.log("val accuracy", self.val_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        return SGD(self.classifier.parameters(), self.lr)

# class AlbumentationsDataset(Dataset):
#     def __init__(self, dataset, transform=None):
#         super().__init__()
#         self.dataset = dataset
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, index):
#         item = self.dataset[index]
#         image, label = item
#         image = ToTensor()(image)
#         if self.transform:
#             transformed = self.transform(image=image)
#             image = transformed["image"]
#         return image, label

@hydra.main(config_path="configs", config_name="config")
def myapp(cfg: DictConfig):
    transform = None
    # train_dataset = MNIST(os.path.expanduser("~/datasets/MNIST"), train=True, download=True, transform=ToTensor())
    # val_dataset = MNIST(os.path.expanduser("~/datasets/MNIST"), train=False, download=True, transform=ToTensor())
    train_dataset = Spiral()
    val_dataset = Spiral()
    # classifier = nn.Sequential(
    #                         nn.Conv2d(1, cfg.channels, (3,3)),
    #                         nn.Flatten(),
    #                         nn.LazyLinear(10)
    #                         )
    regressor = nn.Sequential(
                            nn.Linear(2, cfg.channels),
                            nn.Linear(cfg.channels, cfg.channels),
                            nn.Linear(cfg.channels, 2),
                            nn.Linear(2, 1),
                            )
    # model = RegularizedMnistClassifier(classifier, cfg)
    model = RegularizedMnistClassifier(regressor, cfg)
    trainer = pl.Trainer(logger=TensorBoardLogger("tb_logs"), enable_progress_bar=True, max_epochs=2)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=8)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    classifier_plot(model, train_dataset)

if __name__ == "__main__":
    
    myapp()