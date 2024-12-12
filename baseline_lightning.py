#!/usr/bin/env python3
#
# Code modified from PA3-template.py provided by Sanmukh Kuppannagar for CSDS 451
# updated for homework to use lightning
# changed to accommodate the pytorch resnet50 model

import os
import time
import argparse
import torch
#from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.utils as tu
import torchvision as tv
import torch.optim as optim
import pytorch_lightning as L # convention... not very pythonic

import torchvision as tv
import torchvision.models as models



class L_Net(L.LightningModule):
    """ Class to utilize the same kind of operations as before, but with
    Pytorch Lightning.
    
    Based on tutorial https://lightning.ai/docs/pytorch/stable/starter/converting.html

    Basically, it adds some kind of encoder
    """
    def __init__(self, encoder=None, batch_size=8):
        
        #super(L_Net, self).__init__()
        super().__init__()

        # based on the dev tutorial recomendations
        # here https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html
        # for resnet50 applied to imagenet, start with default weights
        # their example was applied to CIFAR-10, but labled for imagenet... so...
        # that's not a red flag I hope
        self.backbone = models.resnet50(weights="DEFAULT")
        num_filters = self.backbone.fc.in_features
        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

        self.batch_size = batch_size
        self.encoder = encoder
        self.automatic_optimization = False

    def forward(self, x):
        return self.backbone(x)

    def get_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        '''Method to run training step
        '''
        # previously in loop
        inputs, labels = batch

        # still not sure if I need an encoder at all
        # parts of the tutorial make it seem non-optional...
        if self.encoder is not None:
            encoded_inputs = self.encoder(inputs)
        else:
            encoded_inputs = inputs
        
        # pulled from the loop
        outputs = self(encoded_inputs)

        # okay, this didn't turn up any running processes
        #os.system("nvidia-smi")

        # pulled this from the loop as well
        loss = nn.functional.cross_entropy(outputs, labels)

        # code stack exchange said to add to avoid error
        self.manual_backward(loss)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


        return loss

    def manual_backward(self, loss):
        loss.backward()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       second_order_closure, on_tpu, using_native_amp,
                       using_lbfgs):
        optimizer.step()
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # every tutorial has a different optimizer configuration and it's honestly
        # very frustrating

         # return optimizer
        return [optimizer], [lr_scheduler]

class CIFARDataModule(L.LightningDataModule):

    def __init__(self, batch_size=100):
        self.batch_size=batch_size

    # more methods because the lightning intro video was so compelling
    def prepare_data(self):
        
        transform = tv.transforms.Compose([tv.transforms.ToTensor()])
        training_data   = tv.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    def setup(self, stage):
        # do we need a transformer? this was in the pytorch tutorial
        transform = tv.transforms.Compose([tv.transforms.ToTensor()])

        # You should implement these for CIFAR-10. HINT: The dataset may be accessed with Torchvision.
        self.training_data = tv.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    def train_dataloader(self):
        training_loader = tu.data.DataLoader(self.training_data, batch_size=self.batch_size) 
        return training_loader

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=8, type=int, help="specify batch size")
    parser.add_argument("-e", "--epochs", default=1, type=int, help="specify number of training epochs")
    parser.add_argument("-v", "--verbose", action="store_true", help="specify text output")
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    batch_size  = args.batch_size
    epochs      = args.epochs
    device      = args.devices

    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    training_data   = tv.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    training_loader = tu.data.DataLoader(training_data, batch_size=batch_size,
                                         shuffle=True, num_workers=1)

    #training_data = CIFARDataModule(batch_size=batch_size)
    # import the pre-built pytorch module
    # https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/

    net = L_Net(batch_size=batch_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.verbose: print("The device you are using is: ", device)
    net.to(device = device)

    trainer = L.Trainer(accelerator='gpu', devices=2, max_epochs=epochs)

    start = time.time()
    trainer.fit(net, training_loader, training_data)
    end = time.time()
    
    # This is just here to let you know that your model has finished training.
    print(f"Finished training. Used {epochs} epochs with {batch_size} samples per batch, which took {end-start} seconds.")


if __name__=="__main__":
    main()
