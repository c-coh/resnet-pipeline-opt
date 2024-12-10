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

    # batch size
    batch_size = 4
    num_workers = 1
    epochs = 2

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
    
    mpi.init()
    comm = mpi.comm_world()

    #get 'dimensions' of world
    size = comm.Get_size()
    rank = comm.Get_rank()

    #calculate number of layers per GPU
    subsection_size = (len(layers) + (size - 1))// size
    if(rank > 0):
       subsection = layers[(rank - 1)*subsection_size : min(len(layers), (rank)*subsection_size)]

    #device = ("cuda" if torch.cuda.is_available else "cpu")
    #print("The device you are using is: ", device)
    #model.to(device = device)

    start_time = time.time
    for epoch in range(epochs):
        running_loss = 0.0

        # iterate over data for each epoch
        for i, data in enumerate(iterable = training_loader, start = 0):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #access layers of the model
            layers = list(model.children())

            # for a node number
            node_num = 1
            if node_num == 1:
                print('yay')

            #for child in model.children():
            #    grandchildren = 0
            #    for grandkid in child.children():
            #        grandchildren += 1
            #    print(grandchildren)

            

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