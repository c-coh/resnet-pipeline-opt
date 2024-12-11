#!/usr/bin/env python3

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as tu
import torchvision as tv
import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # Download data if needed
        tv.datasets.CIFAR10(root="data", train=True, download=True)

    def setup(self, stage=None):
        # Transformations
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if stage == 'fit' or stage is None:
            self.cifar10_train = tv.datasets.CIFAR10(root="data", train=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

class ResNet50Pipeline(pl.LightningModule):
    def __init__(self, num_classes=10, lr=0.01):
        super().__init__()
        self.lr = lr

        #get resnet model
        resnet50 = models.resnet50(pretrained=False)
        resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)

        #get number of available gpus
        num_gpus = torch.cuda.device_count()

        #divide model into sublayers based on the number of GPUs
        layers = list(resnet50.children())
        subsection_size = len(layers) // num_gpus
        subsection = [nn.Sequential(*layers[i * subsection_size:(i + 1) * subsection_size]) for i in range(num_gpus)]

        #wrap stages into Pipe
        self.pipeline = torch.distributed.pipeline.sync.Pipe(
            nn.Sequential(*subsection),
            chunks=2,
            devices=[f"cuda:{i}" for i in range(num_gpus)],
            checkpoint="never"
        )

    def forward(self, x):
        return self.pipeline(x)

    def training_step(self, batch, batch_idx):

        #time forwards pass
        start_time = time.time()
        x, y = batch
        outputs = self.forward(x)
        forward_time = time.time() - start_time

        loss = nn.CrossEntropyLoss()(outputs, y)

        #time backwards pass
        start_time = time.time()
        self.manual_backward(loss)
        backward_time = time.time() - start_time

        self.log("train_loss", loss)
        self.log("forward_time", forward_time, prog_bar=True)
        self.log("backward_time", backward_time, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=4, type=int, help="specify batch size")
    parser.add_argument("-e", "--epochs", default=2, type=int, help="specify number of training epochs")
    parser.add_argument("-g", "--gpus", default=3, type=int, help="specify number of gpus")
    parser.add_argument("-v", "--verbose", action="store_true", help="specify text output")

    args = parser.parse_args()

    num_workers = 1
    batch_size  = args.batch_size
    epochs      = args.epochs
    gpus        = args.gpus


    init_process_group("nccl")

    #initalize data
    data = CIFAR10Data(batch_size=128)

    #initialize model
    model = ResNet50Pipeline()

    #train the model
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="gpu",
        devices=gpus,
        max_epochs=10
    )
    trainer.fit(model, datamodule=data)
