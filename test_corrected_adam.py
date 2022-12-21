import os

import torch
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

from self_normalizing_nn import self_normalizing_nn_init

from corrected_adam import CorrectedAdam
from SAG_optimizer import SAG, SAG_NoCurvature

class MnistClassifier(pl.LightningModule):
    def __init__(self, model: str ,optimizer_debug_logs=False, sgd=False, no_curvature=False) -> None:
        super().__init__()
        self.accuracy = Accuracy(num_classes=10)
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = 0.001
        self.optimizer_debug_logs = optimizer_debug_logs
        self.sgd = sgd
        self.no_curvature = no_curvature

        if model == "cnn_small":
            # for digits mnist
            self_normalizing = True
            def block(in_dim, out_dim, stride=1):
                conv = torch.nn.Conv2d(in_dim, out_dim, 3, stride, 1)
                if self_normalizing:
                    self_normalizing_nn_init(conv)
                else:
                    torch.nn.init.xavier_uniform_(conv.weight, gain=torch.nn.init.calculate_gain("relu"))
                    torch.nn.init.zeros_(conv.bias)
                return torch.nn.Sequential(conv, 
                                           torch.nn.Identity() if self_normalizing else torch.nn.BatchNorm2d(out_dim), 
                                           torch.nn.SELU() if self_normalizing else torch.nn.ReLU())

            layers = [(1, 16, 1), (16, 16, 1), (16, 32, 2), (32, 32, 2), (32, 64, 2)]
            middle = torch.nn.Sequential(*[block(_id, _od, s) for _id, _od, s in layers])
            end = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1,1)),
                                      torch.nn.Flatten(),
                                      torch.nn.Linear(64, 10))
            torch.nn.init.xavier_uniform_(end[2].weight, gain=torch.nn.init.calculate_gain("relu"))
            torch.nn.init.zeros_(end[2].bias)
            self.model = torch.nn.Sequential(middle, end)

        elif model == "fc":
            # for digits mnist
            def fclayer(*args, **kwargs):
                l = torch.nn.Linear(**kwargs)
                torch.nn.init.xavier_uniform_(l.weight, gain=torch.nn.init.calculate_gain("relu"))
                if l.bias is not None:
                    torch.nn.init.zeros_(l.bias)
                return l
            self.model = torch.nn.Sequential(torch.nn.Flatten(), 
                                             fclayer(in_features=32**2, out_features=500), torch.nn.ReLU(),
                                             fclayer(in_features=500, out_features=300), torch.nn.ReLU(),
                                             fclayer(in_features=300, out_features=10, bias=False))
            
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

    def configure_optimizers(self):
        # return CorrectedAdam(self.model.parameters(), self.lr, (0.9, 0.3), eps=0.1)
        if self.sgd:
            return SGD(self.model.parameters(), self.lr, 0.9, weight_decay=0.0001)
        if self.no_curvature:
            return SAG_NoCurvature(self.model.parameters(), self.lr, (0.3, 0.5), tau=3)
        return SAG(self.model.parameters(), self.lr, (0.3, 0.9), tau=3)

    def forward(self, inputs):
        return self.model(inputs)
    
    def on_after_backward(self):
        # print("on_after_backward")
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}.grad = {param.grad.norm()}")
        ...
    
    def log_optimizer_params(self):
        if not self.optimizer_debug_logs:
            return
        optimizer = self.optimizers(False)
        param_groups = optimizer.param_groups
        logger: TensorBoardLogger = self.logger
        writer: SummaryWriter = logger.experiment
        for group in param_groups:
            for p in group:
                p: torch.nn.parameter.Parameter
                for idx, p in enumerate(optimizer.state):
                    state = optimizer.state[p]
                    ema_avg = state['ema_avg']
                    ema_var = state['ema_var']
                    ema_s_var = state['ema_s_var']
                    curvature = state['curvature']
                    adaptation_factor = state['adaptation_factor']
                    writer.add_histogram(f"ema_avg/{idx}", ema_avg, self.global_step)
                    writer.add_histogram(f"ema_var/{idx}", ema_var, self.global_step)
                    writer.add_histogram(f"ema_s_var/{idx}", ema_s_var, self.global_step)
                    writer.add_histogram(f"curvature/{idx}", curvature, self.global_step)
                    writer.add_histogram(f"adaptation_factor/{idx}", adaptation_factor, self.global_step)

    def training_step(self, batch, batch_idx):
        x, label = batch
        y = self(x)
        loss = F.cross_entropy(y, label)
        self.log("train/loss", loss)
        
        self.log_optimizer_params()
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, label = batch
        y = self(x)
        loss = F.cross_entropy(y, label)
        self.log("val/loss", loss)
        self.accuracy(y, label)
        self.log("val/accuracy", self.accuracy.compute())


def get_dataloaders(dtst):
    if dtst =="fashion_mnist":
        transform = Compose([ToTensor(), Normalize((0), (256))])
        dataset_location = "Fashion_MNIST_data"
        train_dataset = FashionMNIST(root=os.path.join(os.getcwd(), dataset_location), transform=transform, train=True, download=True)
        val_dataset = FashionMNIST(root=os.path.join(os.getcwd(), dataset_location), transform=transform, train=False, download=True)
        
    elif dtst =="mnist":
        transform = Compose([Resize((32, 32)), ToTensor(), Normalize((0), (256))])
        dataset_location = "MNIST_data"
        train_dataset = MNIST(root=os.path.join(os.getcwd(), dataset_location), transform=transform, train=True, download=True)
        val_dataset = MNIST(root=os.path.join(os.getcwd(), dataset_location), transform=transform, train=False, download=True)
        
    
    train_loader = DataLoader(train_dataset, 64, True, num_workers=2)
    val_loader = DataLoader(val_dataset, 64, False, num_workers=2)
    return train_loader, val_loader

def train(dtst: str, model: str, optimizer_debug_logs=False, ckpt: str=None, sgd=False, no_curvature=False):
    train_loader, val_loader = get_dataloaders(dtst)
    if no_curvature:
        optim_name = "_sag_nc"
    elif sgd:
        optim_name = ""
    else:
        optim_name = "_sag"
    name = f"{model}_{dtst}{optim_name}"
    model = MnistClassifier(model, optimizer_debug_logs=optimizer_debug_logs, sgd=sgd, no_curvature=no_curvature)
    trainer = pl.Trainer(logger=TensorBoardLogger(name),
                         max_epochs=300)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt)

if __name__ == "__main__":
    # ckpt for resuming or checking logs
    # ckpt = "/home/lior/experiments/mnist_sag/lightning_logs/version_0/checkpoints/epoch=99-step=93800.ckpt"
    # ckpt = "/home/lior/experiments/mnist_sag/lightning_logs/version_2/checkpoints/epoch=154-step=145390.ckpt"
    # ckpt = "/home/lior/experiments/mnist/lightning_logs/version_0/checkpoints/epoch=36-step=34706.ckpt"
    dtst = "mnist"
    model = "cnn_small"
    # model = "fc"
    ckpt=None
    optimizer_debug_logs = False
    sgd = False
    no_curvature = True
    train(dtst, model, optimizer_debug_logs, ckpt, sgd, no_curvature)