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
    batch_size  = 8
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
        subsection_layers = layers[(my_rank - 1) * subsection_size : min(len(layers), my_rank * subsection_size)]
        my_subsection = nn.Sequential(*subsection_layers)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params = model.parameters(), lr = 0.001)

##### PART 2: TRAINING #####
    if my_rank == 0:
        print(f"training {my_rank}!") if debug else None

        #broadcast how many rounds of batches each node will have to process
        num_mini_batches = len(training_loader)
        print(f"batches: {num_mini_batches}") if debug else None
        
        comm.bcast(num_mini_batches, root=0)
        
        device = ("cuda" if torch.cuda.is_available else "cpu")
        print("The device you are using is: ", device) if debug else None


        #send batches to rank 1
        for i, batch in enumerate(iterable = training_loader, start = 0):

            #loading!
            loading(i, my_rank) if debug else None

            #send input to first GPU in pipeline
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)


            if i % 200 == 0:
                print("IN")
                print(f"labels: {labels}")
                print(f"inputs:{inputs}")
                print()



            comm.send(batch, dest=(my_rank + 1), tag=(my_rank + 1)) #TODO: do I really need tags?

            #once enough data has been sent, start recieving model output
            if i >= world_size:
                #TODO: Figure out what to do with the loss
                loss = comm.recv(source=(world_size - 1), tag=my_rank)
                # Save the received loss periodically
                #torch.save(loss, f'loss_checkpoint_batch_{batch}.pt')
        
        print("done with forward pass!")

    elif my_rank > 0 and my_rank < (world_size - 1):
        print(f"training {my_rank}!") if debug else None

        #recieve number of batches that will be passing through nodes
        num_mini_batches = None
        num_mini_batches = comm.bcast(num_mini_batches, root=0)

        #pass each mini-batch through assigned layers
        for i in range(num_mini_batches):

            #loading!
            loading(i, my_rank) if debug else None


            #send and recieve model data
            data = comm.recv(source=(my_rank - 1), tag= my_rank)
            result = my_subsection(data[0]).squeeze()


            if i % 200 == 0:
                print("MID")
                print(f"labels: {data[1]}")
                print(f"outputs:{result}")
                print()

            comm.send([result, data[1]], dest = (my_rank + 1), tag = (my_rank + 1))
    else:
        num_mini_batches = None
        num_mini_batches = comm.bcast(num_mini_batches, root=0)

        for i in range(num_mini_batches):
            
            #loading!
            loading(i, my_rank) if debug else None

            data = comm.recv(source=(my_rank - 1), tag= my_rank)
            result = my_subsection(data[0]).squeeze()
            labels = data[1]

            #compute loss in final layer
            if i % 200 == 0:
                print("OUT")
                print(f"labels: {labels}")
                print(f"outputs:{result}")
                print()

            loss = criterion(result, labels)
            comm.send(loss, dest = 0, tag = 0)




##### PART 3: BACKPROPOGATION #####
    if my_rank == 0:
        if len(losses) != num_mini_batches:
            print(f"Error: mismatch between num losses ({losses}) and num batches({batches})")

        for i, loss in enumerate(losses):

            #collect the final straggler outputs before starting backpropogation (after all inputs have been fed through)
            if i < world_size:
                loss = comm.recv(source=(world_size - 1), tag=my_rank)
            
            comm.send(loss, dest = (world_size - 1), tag=(world_size-1))

    elif my_rank > 0 and my_rank < (world_size - 1):

        for i in range(num_mini_batches):

            #loading!
            loading(i, my_rank) if debug else None

            #send and recieve model data
            data = comm.recv(source=(my_rank + 1), tag=my_rank)
            result = backwards(data)
            comm.send(result, dest =(my_rank - 1), tag =(my_rank - 1))


    else:

        for i in range(num_mini_batches):
            
            #loading!
            loading(i, my_rank) if debug else None

            loss = comm.recv(source=0, tag= my_rank)
            loss.backwards(data)

            #compute loss in final layer
            outputs, labels = result
            gradient = criterion(outputs, labels)
            comm.send(gradient, dest =(my_rank - 1), tag =(my_rank - 1))



''' 
            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 20  == 19:
                print(f'[{epoch + 1}, {i + 1:5d}] loss : {running_loss / 20:.3f}')

    end_time = time.time

'''
    

if __name__ == "__main__": 
    main()
