import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from torch.nn import functional as F
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import RandomResizedCrop, Compose, ToTensor, Normalize
import hydra
from omegaconf import DictConfig
import os
import os.path
from torch.utils.data import DataLoader, Subset
from typing import Tuple

from pytorch_lightning.loggers import TensorBoardLogger

def get_nn(res, layers, channels):
    return nn.Sequential(nn.Flatten(), nn.Linear(res[0]*res[1], channels), nn.GELU(), 
                         *[nn.Sequential(nn.Linear(channels, channels), nn.GELU()) for _ in range(layers)],
                         nn.Linear(channels, 1, bias=False))

class TransferLearning(pl.LightningModule):
    def __init__(self, teacher: nn.Module, student: nn.Module, cfg):
        super().__init__()
        self.teacher: nn.Module = teacher
        self.student: nn.Module = student
        self.lr = cfg.lr
        self.train_accuracy = torchmetrics.Accuracy(num_classes=1)
        self.val_accuracy = torchmetrics.Accuracy(num_classes=1)

    def forward(self, inputs):
        return self.student(inputs).flatten(), self.teacher(inputs).flatten()
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds_teacher, preds_student = self(inputs)
        normalized_preds_teacher = F.sigmoid(preds_teacher).detach()
        loss = F.cross_entropy(preds_student, normalized_preds_teacher)
        self.train_accuracy(preds_teacher.flatten(),preds_student.flatten())
        self.log("loss", loss)
        self.log("accuracy", self.train_accuracy, prog_bar=True)
        return loss
    
    # def training_epoch_end(self, outputs):
    #     tensorboard = self.logger.experiment
    #     dataset = self.trainer.train_dataloader.dataset.datasets
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.float()
        preds: torch.Tensor = self(inputs)
        loss = F.cross_entropy(preds, labels)
        self.val_accuracy(preds,labels)
        self.log("val loss", loss, prog_bar=True)
        self.log("val accuracy", self.val_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        return SGD(self.student.parameters(), self.lr, momentum=0.1)

class MnistClassifier(pl.LightningModule):
    def __init__(self, classifier, cfg):
        super().__init__()
        self.classifier: nn.Module = classifier
        self.lr = cfg.lr
        self.train_accuracy = torchmetrics.Accuracy(num_classes=1)
        self.val_accuracy = torchmetrics.Accuracy(num_classes=1)

    def forward(self, inputs):
        return self.classifier(inputs).flatten()
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.float()
        preds = self(inputs)
        loss = F.cross_entropy(preds, labels)
        self.train_accuracy(preds,labels)
        self.log("loss", loss)
        self.log("accuracy", self.train_accuracy, prog_bar=True)
        return loss
    
    # def training_epoch_end(self, outputs):
    #     tensorboard = self.logger.experiment
    #     dataset = self.trainer.train_dataloader.dataset.datasets
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.float()
        preds: torch.Tensor = self(inputs)
        loss = F.cross_entropy(preds, labels)
        self.val_accuracy(preds,labels)
        self.log("val loss", loss, prog_bar=True)
        self.log("val accuracy", self.val_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        return SGD(self.classifier.parameters(), self.lr, momentum=0.1)

def get_dataloaders(res: Tuple[int]):
    transform = Compose([RandomResizedCrop((res[0], res[1]), (0.8, 1)), ToTensor(), Normalize((0), (256))])
    train_dataset = MNIST(root=os.path.join(os.getcwd(), "MNIST_data"), transform=transform, train=True, download=True)
    val_dataset = MNIST(root=os.path.join(os.getcwd(), "MNIST_data"), transform=transform, train=False, download=True)
    train_loader = DataLoader(train_dataset, 64, True, num_workers=6)
    val_loader = DataLoader(val_dataset, 64, False, num_workers=6)
    return train_loader, val_loader

def train_initial_model(cfg: DictConfig):
    res = cfg.res
    train_loader, val_loader = get_dataloaders(res)
    channels = cfg.init_channels
    layers = cfg.init_layers
    classifier = get_nn(res, layers, channels)
    model = MnistClassifier(classifier, cfg)
    trainer = pl.Trainer(logger=TensorBoardLogger("transfer_mnist_initial"), 
                         max_epochs=100, 
                         default_root_dir=os.path.join(os.getcwd(), "outputs", "transfer_mnist_initial"))
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return model.classifier

def transfer_to_model(initial_classifier: nn.Module, cfg: DictConfig):
    res = cfg.res
    train_loader, val_loader = get_dataloaders(res)
    teacher = initial_classifier
    for channels in range(1, cfg.init_channels+1, 1):
        for layers in range(1, cfg.init_layers+1, 1):
            print(f"transferring from {(cfg.init_channels, cfg.init_layers)} to {(channels, layers)} (channels, layers)")
            student = get_nn(res, layers, channels)
            # do transfer learning so that the student predicts the output of the teacher
            model = TransferLearning(teacher, student, cfg)
            trainer = pl.Trainer(logger=TensorBoardLogger("transfer_mnist"), 
                                 max_epochs=100, 
                                 default_root_dir=os.path.join(os.getcwd(), "outputs", f"transfer_mnist"))
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def start_experiment(cfg: DictConfig):
    print("training initial model")
    teacher = train_initial_model(cfg)
    
    transfer_to_model(teacher, cfg)
    print("DONE")