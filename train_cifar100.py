""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.resnet import resnet18, resnet50 
from utils.utils import train, test, accuracy, AverageMeter
import wandb

wandb.login()
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--model', type=str, default='resnet18')
    args = parser.parse_args()

    model = resnet18(num_classes=100) if args.model == 'resnet18' else resnet50(num_classes=100)
    model = model.to(device)

    run = wandb.init(
        # Set the project where this run will be logged
        project="radioactive",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
    })

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])

    trainset = datasets.CIFAR100(root='experiments/datasets', train=True, download=True, 
                        transform=transform_train)
    testset = datasets.CIFAR100(root='experiments/datasets', train=False, transform=transform_test)
    trainloader = DataLoader(
        trainset, num_workers=8, batch_size=args.batch_size, shuffle=True)

    testloader = DataLoader(
        testset, num_workers=8, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    iter_per_epoch = len(trainloader)

    lr_schedule = np.interp(np.arange((args.epochs+1) * iter_per_epoch),
                    [0, 5 * iter_per_epoch, args.epochs * iter_per_epoch], [0, 1, 0])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    checkpoint_path = f"{args.model}.pth"

    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, scheduler, device)
        test_loss, test_acc = test(testloader, model, criterion, device)
        print(f'Epoch {epoch + 1} Train Loss {train_loss:.4f} ' + 
            f'Test Loss {test_loss:.4f} Train Acc {train_acc:.4f} Test Acc {test_acc:.4f}')
        wandb.log({"train_acc": train_acc, "train_loss": train_loss})
        wandb.log({"test_acc": test_acc, "test_loss": test_loss})
        
        # Save the best model
        is_best = test_acc > best_acc
        if epoch > 60 and is_best:
            torch.save(model.state_dict(), checkpoint_path)
        best_acc = max(test_acc, best_acc)

    print(f'Best acc: {best_acc}')