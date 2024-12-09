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

### PARSE ARG OPTIONS ###
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=4, type=int, help="specify batch size")
    parser.add_argument("-e", "--epochs", default=2, type=int, help="specify number of training epochs")
    parser.add_argument("-v", "--verbose", action="store_true", help="specify text output")
    args = parser.parse_args()

    num_workers = 1
    batch_size  = args.batch_size
    epochs      = args.epochs


##### PART 0: GET MPI SETTINGS/PARAMETERS #####
    comm = MPI.COMM_WORLD

    #get 'dimensions' of world
    world_size = comm.Get_size()
    my_rank = comm.Get_rank()

    #test if MPI is working properly
    if my_rank == 0:
        print(f"world size is: {world_size}")
    
    print(f"my rank is: {my_rank}")



##### PART 1: INITIAL SETUP OF DATA AND MODEL #####

    ### DATA SETUP FOR WORLD NODE ###
    if(my_rank == 0):
        transform = tv.transforms.Compose([tv.transforms.ToTensor()])

        #compute size of mini-batches
        mini_batch_size = max(1, batch_size// (world_size - 1))

        if batch_size% (world_size - 1):
            print(f"ERROR: batch size of {batch_size} is not divisible by number of nodes { (world_size - 1)}. Defaulting to batch size {mini_batch_size*(world_size - 1)} ")

        #initialize and format data
        training_data = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        training_loader = tu.data.DataLoader(training_data, batch_size=mini_batch_size, shuffle=True, num_workers=num_workers)

        #determine batches that will be passed through
        num_mini_batches = len(training_loader)

    ### MODEL AND PIPELINING SETUP FOR PIPELINING NODES ###
    else:
        #choose your model (resnet50)
        model = models.resnet50()

        # modify last layer to match 10 classes in CIFAR-10
        model.fc = nn.Linear(model.fc.in_features, 10)

        layers = list(model.children())

        #calculate subsection size (number of layers per GPU)
        subsection_size = (len(layers) + (world_size - 1))// world_size

        #assign subsection of layers based on rank
        my_subsection = layers[(my_rank - 1)*subsection_size : min(len(layers), (my_rank)*subsection_size)]

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params = model.parameters(), lr = 0.001)


##### PART 2: TRAINING #####
    if my_rank == 0:
        #broadcast how many rounds of batches each node will have to process
        comm.broadcast(num_mini_batches, root=0)

        #send batches to rank 1
        for batch in range(0, num_mini_batches + world_size):
            
            if batch < num_mini_batches:
                comm.send(training_loader[batch], dest = (my_rank + 1), tag = (my_rank + 1))

            if batch > world_size:
                #TODO: What to do with output?
                comm.recv(output, dest = (my_rank + 1), tag = (my_rank + 1))


        #wait to recieve that number of batches back


    elif my_rank > 0 and my_rank < (mpi_world - 1):
        #recieve number of batches that will be passing through nodes
        num_mini_batches = comm.broadcast(num_mini_batches, root=0)

        #pass each mini-batch through assigned layers
        for _ in range(num_mini_batches):
            data = comm.recv(source=(my_rank - 1), tag= my_rank)
            result = my_subsection(data)
            comm.send(result, dest = (my_rank + 1), tag = (my_rank + 1))
    else:

        for _ in range(num_mini_batches):
            data = comm.recv(source=(my_rank - 1), tag= my_rank)
            result = my_subsection(data)
            comm.sendbuf(result, dest = (my_rank + 1), tag = (my_rank + 1))

            

#TODO: confirm that this properly sets up destination so that last GPU sends to world
            
#TODO: Should the last GPU send data to a buffer? 



##### PART 3: BACKPROPOGATION #####
    if my_rank == 0:
        #feed batches back into 


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