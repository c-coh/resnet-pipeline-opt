#!/usr/bin/env python3

import time
import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as tu

#import pytorch_lighting as pl 

import torchvision as tv
import torchvision.models as models

import mpi4py as mpi
from mpi4py import MPI

#debug function that prints to terminal to show loading process!
def loading(iteration, rank):
    if iteration % 50 == 0:
        message = '\r' + 'processing' + '. '* (rank + 1)
        sys.stdout.write('\r' + ' ' * 20)
        sys.stdout.write(message)
        sys.stdout.flush()

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

### PARSE ARG OPTIONS ###
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=4, type=int, help="specify batch size")
    parser.add_argument("-e", "--epochs", default=2, type=int, help="specify number of training epochs")
    parser.add_argument("-v", "--verbose", action="store_true", help="specify text output")
    args = parser.parse_args()

    #num_workers = 1
    #batch_size  = args.batch_size
    #epochs      = args.epochs

    num_workers = 1
    batch_size  = 32
    mini_batch_size = 4
    epochs      = args.epochs
    debug = True



##### PART 0: GET MPI SETTINGS/PARAMETERS #####
    comm = MPI.COMM_WORLD

    #get 'dimensions' of world
    world_size = comm.Get_size()
    my_rank = comm.Get_rank()


    # DEBUG MODE: test if MPI is working properly
    print(f"world size is: {world_size}") if debug else None
    print(f"my rank is: {my_rank}") if debug else None

##### PART 1: INITIAL SETUP OF DATA AND MODEL #####

    device = ("cuda" if torch.cuda.is_available else "cpu")
    print("The device you are using is: ", device) if debug else None


    ### DATA SETUP FOR WORLD NODE ###
    if(my_rank == 0):
        transform = tv.transforms.Compose([tv.transforms.ToTensor()])

        #initialize and format data
        training_data = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        training_loader = tu.data.DataLoader(training_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        data_iter = iter(training_loader)

        #broadcast how many rounds of batches each node will have to process
        #compute size of mini-batches
        num_mini_batches = batch_size//mini_batch_size
        num_batches = len(training_loader)

        if batch_size%mini_batch_size != 0: 
            print(f"error! batch size {batch_size} is not divisible by mini batch size {mini_batch_size}")
        
        print(f"num of mini batches: {num_mini_batches}") if debug else None
        
        comm.bcast(num_mini_batches, root=0)
        comm.bcast(num_batches, root=0)



    ### MODEL AND PIPELINING SETUP FOR PIPELINING NODES ###
    else:
        #recieve num_mini_batches
        num_mini_batches = 0
        num_batches = 0
        num_mini_batches = comm.bcast(num_mini_batches, root=0)
        num_batches = comm.bcast(num_batches, root=0)


        #choose your model (resnet50)
        model = models.resnet50()

        optimizer = optim.Adam(params = model.parameters(), lr = 0.001)

        # modify last layer to match 10 classes in CIFAR-10
        model.fc = nn.Linear(model.fc.in_features, 10)
    
        optimizer = optim.Adam(params = model.parameters(), lr = 0.001)
        criterion = nn.CrossEntropyLoss()


        layers = list(model.children())

        #calculate subsection size (number of layers per GPU)
        subsection_size = (len(layers) + (world_size - 1))// world_size

        #assign subsection of layers based on rank
        subsection_layers = layers[(my_rank - 1) * subsection_size : min(len(layers), my_rank * subsection_size)]
        my_subsection = nn.Sequential(*subsection_layers)

        print(f"RANK: {my_rank}")
        for layer in my_subsection:
            print(layer) 

        
##### PART 2: TRAINING #####
    for i in range(num_batches):
        #loading!
        loading(i, my_rank) if debug else None

        ## FORWARD PASS ##
        if my_rank == 0:
            print(f"training {my_rank}!") if debug else None
            #initialize losses array
            losses = []

            #split training data into inputs and labels
            batch_inputs, batch_labels = next(data_iter)

            #send batches to rank 1
            for j in range(0, num_mini_batches):

                #send input to first GPU in pipeline
                inputs = batch_inputs[j*mini_batch_size:j*mini_batch_size+mini_batch_size]
                inputs = inputs.to(device)
                comm.send(inputs, dest=(my_rank + 1), tag=(my_rank + 1)) 
                
            all_labels = []
            
            #format data for backwards pass
            for j in range(num_mini_batches):

                labels = batch_labels[j*mini_batch_size:j*mini_batch_size+mini_batch_size]
                all_labels.append(labels)


        elif my_rank > 0 and my_rank < (world_size - 1):
            print(f"training {my_rank}!") if debug else None

            #maintain a results array for backpropogation step
            results = []

            #pass each mini-batch through assigned layers
            for i in range(num_mini_batches):

                #send and recieve model data
                data = comm.recv(source=(my_rank - 1), tag= my_rank)
                data = data.to(next(my_subsection.parameters()).device)
                result = my_subsection(data)
                results.append(result)

                comm.send(result, dest = (my_rank + 1), tag = (my_rank + 1))

        else:
            results = []
            
            for i in range(num_mini_batches):
                data = comm.recv(source=(my_rank - 1), tag= my_rank)
                result = my_subsection(data)
                results.append(result)
        
        print(f"{my_rank} done with forward pass!") if debug else None

        

    ## BACKWARDS PASS ##
        if my_rank == 0:
            comm.send(all_labels, dest=world_size - 1, tag=world_size - 1)

        elif my_rank > 0 and my_rank < (world_size - 1):
            for i in range(num_mini_batches):

                #compute gradient based on previous input
                prev_grad = comm.recv(source=my_rank + 1, tag=my_rank)

                results[i].backward(prev_grad)
                curr_grad = my_subsection.grad
                comm.send(curr_grad, dest=my_rank - 1, tag=my_rank - 1)

        else:
            labels = comm.recv(source=0, tag=my_rank)

            for i in range(num_mini_batches):

                #compute gradient based on previous inputs

                loss = criterion(results[i].squeeze(), labels[i])

                loss.backward()
                
                #gradient = torch.tensor()
                for param in my_subsection.parameters():
                    print(param.grad.shape)


                    #gradient = torch.cat(param.grad)

                comm.send(gradient, dest=my_rank - 1, tag=my_rank - 1)

        print(f"{my_rank} done with backwards pass!")
        comm.Barrier()




    

if __name__ == "__main__": 
    main()
