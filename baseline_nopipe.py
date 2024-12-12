#!/usr/bin/env python3

import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as tu

import torchvision as tv
import torchvision.models as models

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
    if args.verbose: print("The device you are using is: ", device)
    model.to(device = device)

    epoch_start = time.time()
    for epoch in range(epochs):
        running_loss = 0.0

        # iterate over data for each epoch
        batch_sum   = 0
        forward_sum = 0
        back_sum    = 0
        num_batches = 0
        for i, data in enumerate(iterable = training_loader, start = 0):

            batch_start = time.time()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)

            #if i  == 0:
            #    print(f"labels: {labels}")
            #    print(f"outputs:{outputs}")

            loss = criterion(outputs, labels)

            #access layers of the model
            #layers = list(model.children())

            #for child in model.children():
            #    grandchildren = 0
            #    for grandkid in child.children():
            #        grandchildren += 1
            #    print(grandchildren)            

            # update weights
            forward_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            back_time = time.time()

            num_batches += 1
            batch_sum += back_time-batch_start
            forward_sum += forward_time - batch_start
            back_sum += back_time - forward_time

            # minimizing print statements doesn't seem to decrease runtime and does increase nerves by a lot
            # but comment out for time runs
            if i % 500  == 499 and args.verbose:
                print(f'[{epoch + 1}, {i + 1:5d}] loss : {running_loss / 20:.3f}')
                print(f'[{epoch + 1}, {i + 1:5d}] avg batch time : {batch_sum/num_batches :.3f}')
                print(f'[{epoch + 1}, {i + 1:5d}] avg forward time : {forward_sum/num_batches :.3f}')
                print(f'[{epoch + 1}, {i + 1:5d}] avg backwards time : {back_sum/num_batches :.3f}')

    epoch_end = time.time()

    # last round of print statements, not optional for verbose because it's the whole point
    print(f'[{epoch + 1}, {i + 1:5d}] loss : {running_loss / 20:.3f}')
    print(f'[{epoch + 1}, {i + 1:5d}] avg batch time : {batch_sum/num_batches :.3f}')
    print(f'[{epoch + 1}, {i + 1:5d}] avg forward time : {forward_sum/num_batches :.3f}')
    print(f'[{epoch + 1}, {i + 1:5d}] avg backwards time : {back_sum/num_batches :.3f}')

    print(f'Used {epochs} epochs with a batch size of {batch_size}. It took {epoch_end-epoch_start} seconds.')


if __name__ == "__main__": 
    main()
