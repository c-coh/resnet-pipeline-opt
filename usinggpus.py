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
from mpi4py import MPI

#import global hyperparameter values like batch_size and epochs
import globals

def main():

    comm = MPI.COMM_WORLD

    #get 'dimensions' of world
    world_size = comm.Get_size()
    my_rank = comm.Get_rank()


    #TEST IF MPI IS WORKING PROPERLY
    if my_rank == 0:
        print(f"world size is: {world_size}")
    
    print(f"my rank is: {my_rank}")



    ### SET UP DATA ###
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])

    #compute size of mini-batches
    mini_batch_size = max(1, batch_size//world_size)

    #initialize and format data
    training_data = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    training_loader = tu.data.DataLoader(training_data, batch_size=mini_batch_size, shuffle=True, num_workers=num_workers)

    for i, batch in enumerate(training_loader): 
        if i == 4: 
            break
        print(f"Batch {i+1}:") 
        print(batch)


    ### MODEL AND PIPELINING SETUP ###
    
    #choose your model
    model = models.resnet50()

    # modify last layer to match 10 classes in CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10)

    layers = list(model.children())

    #calculate subsection size (number of layers per GPU)
    subsection_size = (len(layers) + (world_size - 1))// world_size

    #assign subsection of layers based on rank
    if(my_rank > 0):
       subsection = layers[(my_rank - 1)*subsection_size : min(len(layers), (my_rank)*subsection_size)]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = 0.001)


    ### TRAINING ###
    if my_rank == 0:
        #determine batches that will be passed through

        batches = training_data.divide

        #send batches to rank 1
        for batch in batches:
            comm.send(result, dest = (my_rank + 1), tag = (my_rank + 1))


        #wait to recieve that number of batches back


    elif my_rank > 0:
        #recieve number of batches to accept
        num_of_batches = comm.recv(source = 0, tag = 'numbatches')

        for _ in range(num_of_batches):
            comm.recv(source=(my_rank - 1), tag= my_rank)
            result = model(data)

#TODO: confirm that this properly sets up destination so that last GPU sends to world
            comm.send(result, dest = (my_rank + 1)%world_size, tag = (my_rank + 1)%world_size)


'''
    ### BACKPROPOGATION ###
    if my_rank == 0:
        #feed batches back into 

    #device = ("cuda" if torch.cuda.is_available else "cpu")
    #print("The device you are using is: ", device)
    #model.to(device = device)
    '''

    ''' start_time = time.time
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

    print(f'Used {epochs} epochs with a batch size of {batch_size}. It took {end_time - start_time} seconds.')'''
    

if __name__ == "__main__": 
    main()