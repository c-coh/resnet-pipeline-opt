#!/usr/bin/env python3

import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as tu

import torchvision as tv
import torchvision.models as models

import mpi4py as mpi
from mpi4py import MPI

#debug function that prints to terminal to show loading process!
def loading(iteration, rank):
    pass
    #if iteration % 500 == 0:
    #    message = '\r' + 'processing' + '. '* (rank + 1)
    #    sys.stdout.write('\r' + ' ' * 20)
    #    sys.stdout.write(message)
    #    sys.stdout.flush()

def main():
    # share memory between nodes
    torch.multiprocessing.set_sharing_strategy('file_system')

    ### PARSE ARG OPTIONS ###
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="specify batch size")
    parser.add_argument("-m", "--mini_batch_size", default=4, type=int, help="specify mini batch size")
    parser.add_argument("-e", "--epochs", default=1, type=int, help="specify number of training epochs")
    parser.add_argument("-v", "--verbose", action="store_true", help="specify text output")
    args = parser.parse_args()

    #num_workers = 1
    #batch_size  = args.batch_size
    #epochs      = args.epochs

    num_workers     = 1
    batch_size      = args.batch_size
    mini_batch_size = args.mini_batch_size
    epochs          = args.epochs
    debug = True


    ##### PART 0: GET MPI SETTINGS/PARAMETERS #####
    comm = MPI.COMM_WORLD

    #get 'dimensions' of world
    world_size = comm.Get_size()
    my_rank = comm.Get_rank()


    # DEBUG MODE: test if MPI is working properly
    print(f"world size is: {world_size}") if args.verbose else None
    print(f"my rank is: {my_rank}") if args.verbose else None

    ##### PART 1: INITIAL SETUP OF DATA AND MODEL #####

    device = ("cuda" if torch.cuda.is_available else "cpu")
    print("The device you are using is: ", device) if args.verbose else None


    ### DATA SETUP FOR WORLD NODE ###

    if(my_rank == 0):
        epoch_start = MPI.Wtime()

        # set CIFAR-10 to send tensors through net
        transform = tv.transforms.Compose([tv.transforms.ToTensor()])

        #initialize and format data
        training_data = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        training_loader = tu.data.DataLoader(training_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        data_iter = iter(training_loader)

        #broadcast how many rounds of batches each node will have to process
        #compute size of mini-batches
        num_mini_batches = batch_size//mini_batch_size
        num_batches = len(training_loader)

        # divide batches into chunks
        # check and make sure that it fits evenly
        if batch_size%mini_batch_size != 0: 
            print(f"error! batch size {batch_size} is not divisible by mini batch size {mini_batch_size}")
        
        print(f"num of mini batches: {num_mini_batches}") if args.verbose else None
        

        comm.bcast(num_mini_batches, root=0)
        comm.bcast(num_batches, root=0)

        model = models.resnet50()
        # modify last layer to match 10 classes in CIFAR-10
        model.fc = nn.Linear(model.fc.in_features, 10)

        #for param in model.parameters():
        #    param.requires_grad = True

        # node 0 doesn't run the model
        #optimizer = optim.Adam(params = model.parameters(), lr = 0.001)
        #criterion = nn.CrossEntropyLoss()

        if args.verbose:
            print(f"Node 0 took {MPI.Wtime() - epoch_start} seconds to run setup")

    ### MODEL AND PIPELINING SETUP FOR PIPELINING NODES ###
    else:
        #recieve num_mini_batches
        num_mini_batches = 0
        num_batches = 0
        print(num_mini_batches)

        # link broadcast from node 0
        num_mini_batches = comm.bcast(num_mini_batches, root=0)
        num_batches = comm.bcast(num_batches, root=0)

        if args.verbose:
            print(f"Node zero has broadcast {num_mini_batches} mini batches to node {my_rank}")
            print(f"Node zero has broadcast {num_batches} batches to node {my_rank}")

        # import the pre-built pytorch module
        # https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
        model = models.resnet50()
        # modify last layer to match 10 classes in CIFAR-10
        model.fc = nn.Linear(model.fc.in_features, 10)

        # every node gets these two things as well
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params = model.parameters(), lr = 0.001)

        layers = list(model.children())

        #calculate subsection size (number of layers per GPU)
        subsection_size = (len(layers) + (world_size - 1))// world_size

        #assign subsection of layers based on rank
        subsection_layers = layers[(my_rank - 1) * subsection_size : min(len(layers), my_rank * subsection_size)]
        my_subsection = nn.Sequential(*subsection_layers)
        
        #if args.verbose:
        #    print(f"RANK: {my_rank}, LAYERS:")
        #    for layer in my_subsection:
        #        print(layer) 

        # set every parameter to be optimizable
        #for param in my_subsection.parameters():
        #    param.requires_grad = True

    ##### PART 2: TRAINING #####

    # TODO For epoch in range(epochs)...

    #for i in range(1):
    for i in range(num_batches):
        
        # print diagnostics
        #loading(i, my_rank) if args.verbose else None

        ## FORWARD PASS ## ------------------------------------------------------------------------

        # node zero loads new mini-batches sequentially and inputs them to node 1
        if my_rank == 0:
            
            print(f"training node {my_rank} (should be 0) and loading data") if args.verbose else None
            

            #split training data into inputs and labels
            batch_inputs, batch_labels = next(data_iter)

            #send batches to rank 1
            for j in range(0, num_mini_batches):

                #send input to first GPU in pipeline
                inputs = batch_inputs[j*mini_batch_size:j*mini_batch_size+mini_batch_size]
                inputs = inputs.to(device)
                comm.send(inputs, dest=(my_rank + 1)) 
                #once enough data has been sent, start recieving model output

            if args.verbose:
                print(f"node {my_rank} is done feeding mini-batch into pipeline")

            # HALT!
            print(f"Node {my_rank} hitting barrier") if args.verbose else None
            comm.Barrier()
            print(f"Node {my_rank} escaping barrier") if args.verbose else None

            #print(f"Node {my_rank} recieving result from node {world_size - 1}") if args.verbose else None
            
            #results =comm.recv(source=(world_size - 1), tag= my_rank) 
            #print(f"node {my_rank} recieved results as result form {world_size - 1}") if args.verbose else None

            #initialize lables array
            mini_batch_labels = []

            #format data for backwards pass
            #for j in range(num_mini_batches):
            #    labels = batch_labels[j*mini_batch_size:j*mini_batch_size+mini_batch_size]
            #    all_labels.append(labels)

            # get target labels for mini-batch
            for j in range(mini_batch_size):
                label = batch_labels[i*mini_batch_size+j]
                mini_batch_labels.append(label)

            print(len(mini_batch_labels))
            mini_batch_labels = torch.stack(mini_batch_labels)
            

        elif my_rank > 0 and my_rank < (world_size - 1):

            print(f"training {my_rank} (should be >0)") if args.verbose else None

            #maintain a results array for backpropogation step
            outputs = []

            #pass each mini-batch through assigned layers
            for i in range(num_mini_batches):

                #send and recieve model data
                input = comm.recv(source=(my_rank - 1))
                #print(f"Node {my_rank} reciving input") if args.verbose else None

                input = input.to(next(my_subsection.parameters()).device)
                output = my_subsection(input)
                outputs.append(output)

                comm.send(output, dest = (my_rank + 1), tag = i)
                #print(f"Node {my_rank} sending output") if args.verbose else None

            print(f"Node {my_rank} hitting barrier") if args.verbose else None
            comm.Barrier()
            print(f"Node {my_rank} escaping barrier") if args.verbose else None

        else:
            results = []
            
            for i in range(mini_batch_size):

                input = comm.recv(source=(my_rank - 1), tag = i)
                #print(f"Node {my_rank} reciving input") if args.verbose else None

                result = my_subsection(input)
                results.append(result)

                #print(f"Node {my_rank} has computed final result") if args.verbose else None

            print(f"result shape {result.shape}")
            results = torch.stack(results)
            
            print(f"results shape {results.shape}")

            print(f"Node {my_rank} hitting barrier") if args.verbose else None
            comm.Barrier()
            print(f"Node {my_rank} escaping barrier") if args.verbose else None

            #comm.send(results, dest = 0, tag = 0)
            #print(f"Node {my_rank} sending {results}") if args.verbose else None
        
        print(f"node {my_rank} has started backwards pass") if args.verbose else None
        ## BACKWARDS PASS ##
        if my_rank == 0:
            #collect the final straggler outputs before starting backpropogation (after all inputs have been fed through)
            
            # send labels to the last node
            comm.send(mini_batch_labels, dest=world_size - 1)

            model = comm.bcast(model, source=world_size-1)

        elif my_rank > 0 and my_rank < (world_size - 1):

            for i in range(num_mini_batches):

                #compute gradient based on previous input
                prev_grad = comm.recv(source=my_rank + 1)
                print(f"node {my_rank} has recieved previous gradient")

                outputs[i].backward(prev_grad)
                curr_grad = my_subsection.grad

                if my_rank > 1:
                    comm.send(curr_grad, dest=my_rank - 1)

            model = comm.bcast(model, source=world_size-1)
            
            layers = list(model.children())

            #calculate subsection size (number of layers per GPU)
            subsection_size = (len(layers) + (world_size - 1))// world_size

            #assign subsection of layers based on rank
            subsection_layers = layers[(my_rank - 1) * subsection_size : min(len(layers), my_rank * subsection_size)]
            my_subsection = nn.Sequential(*subsection_layers)
        else:

            mini_batch_labels = comm.recv(source=0)


            # try bulk update
            # this is almost definitely wrong
            loss = criterion(results, mini_batch_labels)
            loss.backward()

            print(loss)
            optimizer.step()

            #for i in range(num_mini_batches):

                ##compute gradient based on previous inputs
                
             #   print(results[i].shape)

              #  loss = criterion(results[i].squeeze(), all_labels[i])
               # loss.backward()
                
                #gradient = torch.tensor()
                #for param in my_subsection.parameters():
                #    print(param.grad.shape)
                #    gradient = torch.stack(param.grad)

                #comm.send(gradient, dest=my_rank - 1, tag=my_rank - 1)
            model = comm.bcast(model, source=world_size-1)
            layers = list(model.children())

            #calculate subsection size (number of layers per GPU)
            subsection_size = (len(layers) + (world_size - 1))// world_size

            #assign subsection of layers based on rank
            subsection_layers = layers[(my_rank - 1) * subsection_size : min(len(layers), my_rank * subsection_size)]
            my_subsection = nn.Sequential(*subsection_layers)
            
        print(f"node {my_rank} has finished the backwards pass") if args.verbose else None
        


        
        print(f"Node {my_rank} hitting barrier") if args.verbose else None
        comm.Barrier()
        print(f"Node {my_rank} escaping barrier") if args.verbose else None

        
    

if __name__ == "__main__": 
    main()
