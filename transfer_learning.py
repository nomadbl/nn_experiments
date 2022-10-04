import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics

from torch.nn import functional as F
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import RandomResizedCrop, Compose, ToTensor, Normalize
from omegaconf import DictConfig
import os
import os.path
from torch.utils.data import DataLoader, Subset
from typing import Tuple

from pytorch_lightning.loggers import TensorBoardLogger

def self_normalizing_nn_init(layer: nn.Linear):
    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity="linear")
    if not layer.bias is None:
        nn.init.constant_(layer.bias, 0)
    return layer

class ChannelsView(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, inputs: torch.Tensor):
        b, c, h, w = inputs.shape
        inputs = inputs.moveaxis(1, -1).reshape((-1, c))
        output: torch.Tensor = self.layer(inputs)
        _, c = output.shape
        output = output.reshape((b, h, w, c)).moveaxis(-1, 1)
        return output

class PatchesView(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, inputs: torch.Tensor):
        b, c, h, w = inputs.shape
        inputs = inputs.reshape((-1, h*w))
        output: torch.Tensor = self.layer(inputs)
        _, L = output.shape
        assert L==h*w
        output = output.reshape((b, c, h, w))
        return output

class MixerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_patches):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_patches = n_patches
        self.layer = nn.Sequential(
                            ChannelsView(self_normalizing_nn_init(nn.Linear(in_dim, out_dim))),
                            nn.SELU(),
                            PatchesView(self_normalizing_nn_init(nn.Linear(n_patches, n_patches))),
                            nn.SELU()
                        )
    
    def forward(self, inputs):
        return self.layer(inputs)

def get_mixer_nn(size: Tuple[int], layers, channels):
    size = tuple(size)
    n_patches = size[0] * size[1]
    model = nn.Sequential(nn.Upsample(size), 
                          MixerLayer(1, channels, n_patches), 
                          *[MixerLayer(channels, channels, n_patches) for _ in range(layers)],
                          nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten(),
                          self_normalizing_nn_init(nn.Linear(channels, 10, bias=False))
                         )
    return model

def get_FC_nn(res, layers, channels):
    model = nn.Sequential(nn.Flatten(), 
                          self_normalizing_nn_init(nn.Linear(res[0]*res[1], channels)), nn.SELU(), 
                          *[nn.Sequential(self_normalizing_nn_init(nn.Linear(channels, channels)), nn.SELU()) for _ in range(layers)],
                          self_normalizing_nn_init(nn.Linear(channels, 10, bias=False))
                         )
    return model

class TransferLearning(pl.LightningModule):
    def __init__(self, teacher: nn.Module, student: nn.Module, cfg):
        super().__init__()
        self.teacher: nn.Module = teacher
        self.student: nn.Module = student
        self.lr = cfg.lr
        self.train_accuracy = torchmetrics.Accuracy(num_classes=10)
        self.val_accuracy = torchmetrics.Accuracy(num_classes=10)

    def forward(self, inputs):
        return self.student(inputs), self.teacher(inputs)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds_student, preds_teacher = self(inputs)
        normalized_preds_teacher = F.sigmoid(preds_teacher).detach()
        loss = F.cross_entropy(preds_student, normalized_preds_teacher)
        self.train_accuracy(preds_student, labels)
        self.log("train/loss", loss)
        self.log("train/accuracy", self.train_accuracy, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds_student, preds_teacher = self(inputs)
        normalized_preds_teacher = F.sigmoid(preds_teacher).detach()
        loss = F.cross_entropy(preds_student, normalized_preds_teacher)
        self.val_accuracy(preds_student, labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        return SGD(self.student.parameters(), self.lr, momentum=0.9, weight_decay=0.0001)

class MnistClassifier(pl.LightningModule):
    def __init__(self, classifier, cfg):
        super().__init__()
        self.classifier: nn.Module = classifier
        self.lr = cfg.lr
        self.train_accuracy = torchmetrics.Accuracy(num_classes=10)
        self.val_accuracy = torchmetrics.Accuracy(num_classes=10)

    def forward(self, inputs):
        return self.classifier(inputs)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.long()
        preds = self(inputs)
        loss = F.cross_entropy(preds, labels)
        self.train_accuracy(preds,labels)
        self.log("train/loss", loss)
        self.log("train/accuracy", self.train_accuracy, prog_bar=True)
        return loss
    
    # def on_after_backward(self):
        # for param in self.classifier.parameters():
            # print(param.grad)
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.long()
        preds: torch.Tensor = self(inputs)
        loss = F.cross_entropy(preds, labels)
        self.val_accuracy(preds,labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        return SGD(self.classifier.parameters(), self.lr, momentum=0.9, weight_decay=0.0001)

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
    classifier = get_mixer_nn(res, layers, channels)
    model = MnistClassifier(classifier, cfg)
    trainer = pl.Trainer(logger=TensorBoardLogger("transfer_mnist_initial"), 
                         callbacks=[EarlyStopping(monitor="val/accuracy", mode="max")],
                         max_epochs=100, 
                         default_root_dir=cfg.pretrained_root_path)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return model.classifier

def transfer_to_model(initial_classifier: nn.Module, cfg: DictConfig):
    res = cfg.res
    train_loader, val_loader = get_dataloaders(res)
    teacher = initial_classifier
    for channels in range(5, cfg.init_channels+1, 1):
        for layers in range(2, cfg.init_layers+1, 1):
            print(f"transferring from {(cfg.init_channels, cfg.init_layers)} to {(channels, layers)} (channels, layers)")
            student = get_mixer_nn(res, layers, channels)
            # do transfer learning so that the student predicts the output of the teacher
            model = TransferLearning(teacher, student, cfg)
            trainer = pl.Trainer(logger=TensorBoardLogger("transfer_mnist"),
                                 callbacks=[EarlyStopping(monitor="val/accuracy", mode="max", min_delta=0.05, patience=3, verbose=False)],
                                 max_epochs=100, 
                                 default_root_dir=os.path.join(os.getcwd(), "outputs", f"transfer_mnist"))
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def start_experiment(cfg: DictConfig):
    if os.path.exists(cfg.pretrained_root_path):
        print("loading pretrained intial model")
        res = cfg.res
        channels = cfg.init_channels
        layers = cfg.init_layers
        classifier = get_mixer_nn(res, layers, channels)
        teacher = MnistClassifier.load_from_checkpoint(cfg.pretrained_ckpt, classifier=classifier, cfg=cfg).classifier
    else:
        print("training initial model")
        teacher = train_initial_model(cfg)
    
    transfer_to_model(teacher, cfg)
    print("DONE")

def main():
    # cfg = DictConfig({"lr": 0.01, "init_channels": 100, "init_layers": 4, "res": [64, 64]})
    cfg = DictConfig({"lr": 0.002, "init_channels": 100, "init_layers": 2, "res": [16, 16], 
                      "pretrained_root_path": os.path.join(os.getcwd(), "transfer_mnist_initial"),
                      "pretrained_ckpt": "/home/lior/experiments/transfer_mnist_initial/lightning_logs/version_0/checkpoints/epoch=58-step=55342.ckpt"})
    start_experiment(cfg)

if __name__ == "__main__":
    main()