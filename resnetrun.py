#!/usr/bin/env python3

import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as tu

#import pytorch_lighting as pl 

import torchvision as tv
import torchvision.models as models

import mpi4py as mpi

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=4, type=int, help="specify batch size")
    parser.add_argument("-e", "--epochs", default=2, type=int, help="specify number of training epochs")
    parser.add_argument("-v", "--verbose", action="store_true", help="specify text output")
    args = parser.parse_args()

    num_workers = 1
    batch_size  = args.batch_size
    epochs      = args.epochs

    ### MODEL SETUP ###

    transform = tv.transforms.Compose([tv.transforms.ToTensor()])

    training_data = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    training_loader = tu.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        #choose your model
    model = models.resnet50()

    # modify last layer to match 10 classes in CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = 0.001)
    
    device = ("cuda" if torch.cuda.is_available else "cpu")
    print("The device you are using is: ", device)
    model.to(device = device)

    start_time = time.time
    for epoch in range(epochs):
        running_loss = 0.0

        # iterate over data for each epoch
        for i, data in enumerate(iterable = training_loader, start = 0):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 20  == 19:
                print(f'[{epoch + 1}, {i + 1:5d}] loss : {running_loss / 20:.3f}')

    end_time = time.time

    print(f'Used {epochs} epochs with a batch size of {batch_size}. It took {end_time - start_time} seconds.')

if __name__ == "__main__": 
    main()
