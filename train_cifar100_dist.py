""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime
from tqdm.autonotebook import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.resnet import resnet18, resnet50 
# from utils.utils import train, test, accuracy, AverageMeter
from utils.utils import Timer
from accelerate import Accelerator
import wandb

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accelerator = Accelerator(log_with="wandb")


def train_model(model, train_set_loader, optimizer, scheduler):
    model.train() # For special layers
    total = 0
    correct = 0
    total_loss = 0
    for images, targets in tqdm(train_set_loader, desc="Training"):
        optimizer.zero_grad()

        outputs = model(images)
        loss = F.cross_entropy(outputs, targets, reduction='mean')
        total_loss += torch.sum(loss)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        _, predicted = torch.max(outputs.data, 1)
        accurate_preds = accelerator.gather(predicted) == accelerator.gather(targets)
        total += accurate_preds.shape[0]
        correct += accurate_preds.long().sum()

    average_train_loss = total_loss / total
    accuracy = 100. * correct.item() / total

    return average_train_loss, accuracy

@torch.no_grad()
def test_model(model, test_set_loader):
    model.eval() # For special layers
    total = 0
    correct = 0
    total_loss = 0
    for images, targets in tqdm(test_set_loader, desc="Testing"):
        outputs = model(images)
        loss = F.cross_entropy(outputs, targets, reduction='mean')
        total_loss += torch.sum(loss)

        _, predicted = torch.max(outputs.data, 1)
        # correct += predicted.eq(targets.data).cpu().sum()
        accurate_preds = accelerator.gather(predicted) == accelerator.gather(targets)
        total += accurate_preds.shape[0]
        correct += accurate_preds.long().sum()

    average_test_loss = total_loss / total
    accuracy = 100. * correct.item() / total
    return average_test_loss, accuracy


def main(args):
    accelerator.print("Starting experiment.")
    accelerator.init_trackers(
        project_name="radioactive", 
        config={"epochs": args.epochs, "learning_rate": args.lr}
    )

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
        trainset, batch_size=args.batch_size, shuffle=True)

    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False)
    
    device = accelerator.device
    model = resnet18(num_classes=100) if args.model == 'resnet18' else resnet50(num_classes=100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    iter_per_epoch = len(trainloader)

    lr_schedule = np.interp(np.arange((args.epochs+1) * iter_per_epoch),
                    [0, 5 * iter_per_epoch, args.epochs * iter_per_epoch], [0, 1, 0])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    # use accelerate to automatically handle device placement
    accelerator.print("Preparing model, optimizer, and scheduler.")
    model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, testloader, scheduler
    )
    accelerator.print("Initialization complete.")

    checkpoint_path = f"{args.model}.pth"
    best_acc = 0
    t = Timer()
    for epoch in range(args.epochs):
        t.start()
        # train_loss, train_acc = train(trainloader, model, criterion, optimizer, scheduler, device)
        # test_loss, test_acc = test(testloader, model, criterion, device)
        train_loss, train_acc = train_model(model, trainloader, optimizer, scheduler)
        test_loss, test_acc = test_model(model, testloader)
        
        # Save the best model
        is_best = test_acc > best_acc
        if epoch > 60 and is_best:
            best_acc = test_acc
            # torch.save(model.state_dict(), checkpoint_path)
            accelerator.wait_for_everyone()
            accelerator.print("Saving checkpoint.")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_optimizer = accelerator.unwrap_model(optimizer)
            unwrapped_scheduler = accelerator.unwrap_model(scheduler)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': unwrapped_optimizer.state_dict(),
                'lr_scheduler_state_dict': unwrapped_scheduler.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                }, checkpoint_path)
        elapsed_time = t.stop()
        accelerator.print(f"End of epoch {epoch}, took {elapsed_time:0.4f} seconds.")
        accelerator.print(f"Average Train Loss: {train_loss}")
        accelerator.print(f"Average Test Loss: {test_loss}")
        accelerator.print(f"Top-1 Train Accuracy: {train_acc}")
        accelerator.print(f"Top-1 Test Accuracy: {test_acc}")
        accelerator.log({
            "train_loss": train_loss, 
            "test_loss": test_loss,
            "train_acc": train_acc, 
            "test_acc": test_acc})

    accelerator.print(f'Top-1 Best Test Accuracy: {best_acc}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--model', type=str, default='resnet18')
    args = parser.parse_args()

    main(args)