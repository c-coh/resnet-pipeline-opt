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

        model = models.resnet50()

        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(params = model.parameters(), lr = 0.001)
        criterion = nn.CrossEntropyLoss()


    ### MODEL AND PIPELINING SETUP FOR PIPELINING NODES ###
    else:
        #recieve num_mini_batches
        num_mini_batches = 0
        num_batches = 0
        print(num_mini_batches)
        num_mini_batches = comm.bcast(num_mini_batches, root=0)
        num_batches = comm.bcast(num_batches, root=0)


        #choose your model (resnet50)
        model = models.resnet50()

        optimizer = optim.Adam(params = model.parameters(), lr = 0.001)

        # modify last layer to match 10 classes in CIFAR-10
        model.fc = nn.Linear(model.fc.in_features, 10)

        layers = list(model.children())

        #calculate subsection size (number of layers per GPU)
        subsection_size = (len(layers) + (world_size - 1))// world_size

        #assign subsection of layers based on rank
        subsection_layers = layers[(my_rank - 1) * subsection_size : min(len(layers), my_rank * subsection_size)]
        my_subsection = nn.Sequential(*subsection_layers)
        
        for param in my_subsection.parameters():
            param.requires_grad = True


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
                #once enough data has been sent, start recieving model output
            comm.Barrier()
            result =comm.recv(source=(world_size - 1), tag= my_rank)
            print(result)
            for j, r in enumerate(result):

                labels = batch_labels[j*mini_batch_size:j*mini_batch_size+mini_batch_size]
                print('e')
                print(r)
                print(j)
                loss = criterion(r.squeeze(), labels)
                losses.append(loss)


            print("done with forward pass!")
        elif my_rank > 0 and my_rank < (world_size - 1):
            print(f"training {my_rank}!") if debug else None

            #pass each mini-batch through assigned layers
            for i in range(num_mini_batches):

                #send and recieve model data
                data = comm.recv(source=(my_rank - 1), tag= my_rank)
                data = data.to(next(my_subsection.parameters()).device)
                result = my_subsection(data)

                comm.send(result, dest = (my_rank + 1), tag = (my_rank + 1))
            comm.Barrier()

        else:
            results = []
            
            print("eeee")
            for i in range(num_mini_batches):

                data = comm.recv(source=(my_rank - 1), tag= my_rank)
                result = my_subsection(data)
                results.append(result)
            comm.Barrier()

            comm.send(results, dest = 0, tag = 0)
            print("SENT")
        
        

    ## BACKWARDS PASS ##
        if my_rank == 0:
            #collect the final straggler outputs before starting backpropogation (after all inputs have been fed through)
            print("started")
            for _ in range(num_mini_batches):
                
                #initialize gradient and pass through model
                grad_output = torch.ones_like(losses[0])
                optimizer.zero_grad()
                losses[0].backward(grad_output, retain_graph=True)
                comm.send(grad_output, dest=world_size - 1, tag=world_size - 1)

        elif my_rank > 0 and my_rank < (world_size - 1):
            print("started")

            for _ in range(num_mini_batches):

                #compute gradient based on previous input
                prev_grad = comm.recv(source=my_rank + 1, tag=my_rank + 1)
                result_grad = torch.autograd.grad(outputs=my_subsection.output, inputs=my_subsection.parameters(), grad_outputs=prev_grad)
                comm.send(result_grad, dest=my_rank - 1, tag=my_rank - 1)

        else:
            for _ in range(num_mini_batches):

                #compute gradient based on previous input
                loss_grad = comm.recv(source=0, tag=my_rank)
                loss_grad.backward()
                comm.send(loss_grad, dest=my_rank - 1, tag=my_rank - 1)

        print(f"done with backwards pass!")


    

if __name__ == "__main__": 
    main()