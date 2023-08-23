import os
import re
import shutil

import torch
import torchvision
import torchvision.transforms.transforms as transforms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

from radioactive.dataset_wrappers import MergedDataset
from utils.utils import NORMALIZE_CIFAR100
from utils.utils import Timer

from accelerate import Accelerator
accelerator = Accelerator()

import logging
from utils.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

# Datasets use pickle, so we can't just pass in a lambda
def numpy_loader(x):
    return transforms.ToPILImage()(np.load(x))

def get_data_loaders_cifar(marked_images_directory, augment, batch_size=512, num_workers=1, dataset='cifar10'):

    cifar_dataset_root = "experiments/datasets" # Will download here

    # Base Training Set
    if dataset == 'cifar10':
        base_train_set = torchvision.datasets.CIFAR10(cifar_dataset_root, download=True)
    elif dataset == 'cifar100':
        base_train_set = torchvision.datasets.CIFAR100(cifar_dataset_root, download=True)

    # Load marked data from Numpy img format - no transforms
    extensions = ("npy")
    marked_images = torchvision.datasets.DatasetFolder(marked_images_directory, numpy_loader, extensions=extensions)

    # Setup Merged Training Set: Vanilla -> Merged <- Marked
    # MergedDataset allows you to replace certain examples with marked alternatives
    merge_to_vanilla = [None] * len(marked_images)
    for i, (path, target) in enumerate(marked_images.samples):
        img_id = re.search('[0-9]+', os.path.basename(path))
        merge_to_vanilla[i] = int(img_id[0])

    merged_train_set = MergedDataset(base_train_set, marked_images, merge_to_vanilla)

    # Add Transform and Get Training set dataloader
    transforms_list = []
    if augment:
        transforms_list += [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip()]
    
    transforms_list += [transforms.ToTensor(), NORMALIZE_CIFAR100]
    
    train_transform = transforms.Compose(transforms_list)
    merged_train_set.transform = train_transform

    train_set_loader = torch.utils.data.DataLoader(merged_train_set,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True)

    # Test Set (Simple)
    test_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_CIFAR100])
    if dataset == 'cifar10':
        test_set = torchvision.datasets.CIFAR10(cifar_dataset_root, train=False, transform=test_transform)
    elif dataset == 'cifar100':
        test_set = torchvision.datasets.CIFAR100(cifar_dataset_root, train=False, transform=test_transform)
    test_set_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=batch_size, 
                                                  num_workers=num_workers, 
                                                  shuffle=False,
                                                  pin_memory=True)

    return train_set_loader, test_set_loader


def train_model(model, train_set_loader, optimizer):
    model.train() # For special layers
    total = 0
    correct = 0
    total_loss = 0
    for images, targets in tqdm(train_set_loader, desc="Training"):
        optimizer.zero_grad()

        output = model(images)
        loss = F.cross_entropy(output, targets, reduction='mean')
        total_loss += torch.sum(loss)
        accelerator.backward(loss)
        optimizer.step()
        # logger.info(f"Batch Loss: {loss}")

        _, predicted = torch.max(output.data, 1)
        # correct += predicted.eq(targets.data).cpu().sum()
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
    for images, targets in tqdm(test_set_loader, desc="Testing"):
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        # correct += predicted.eq(targets.data).cpu().sum()
        accurate_preds = accelerator.gather(predicted) == accelerator.gather(targets)
        total += accurate_preds.shape[0]
        correct += accurate_preds.long().sum()

    accuracy = 100. * correct.item() / total
    return accuracy


def main(dataloader_func, model, optimizer_callback, output_directory, tensorboard_log_directory, 
         lr_scheduler=None, epochs=150):

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # Setup regular log file
    logfile_path = os.path.join(output_directory, "logfile.txt")
    setup_logger_tqdm(logfile_path)

    # Setup TensorBoard logging
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)

    # Choose Training Device
    # use_cuda = torch.cuda.is_available()
    # logger.info(f"CUDA Available? {use_cuda}")
    # device = "cuda" if use_cuda else "cpu"   

    # Dataloaders
    train_set_loader, test_set_loader = dataloader_func()

    # Model & Optimizer
    # model.to(device)
    optimizer = optimizer_callback(model)
    if lr_scheduler:
        lr_scheduler = lr_scheduler(optimizer)

    if accelerator.is_local_main_process:
        logger.info(f"Epoch Count: {epochs}")

    # Load Checkpoint
    checkpoint_file_path = os.path.join(output_directory, "checkpoint.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_file_path):
        logger.info("Checkpoint Found - Loading!")

        checkpoint = torch.load(checkpoint_file_path)
        logger.info(f"Last completed epoch: {checkpoint['epoch']}")
        logger.info(f"Average Train Loss: {checkpoint['train_loss']}")
        logger.info(f"Top-1 Train Accuracy: {checkpoint['train_accuracy']}")
        logger.info(f"Top-1 Test Accuracy: {checkpoint['test_accuracy']}")
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resuming at epoch {start_epoch}")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    else:
        logger.info("No checkpoint found, starting from scratch.")

    # Use accelerator
    model, optimizer, train_set_loader, test_set_loader = accelerator.prepare(
        model, optimizer, train_set_loader, test_set_loader
    )

    # Training Loop
    t = Timer()
    best_test = 0
    for epoch in range(start_epoch, epochs):
        t.start()
        if accelerator.is_local_main_process:
            logger.info(f"Commence EPOCH {epoch}")

        # Train
        train_loss, train_accuracy = train_model(model, train_set_loader, optimizer)
        tensorboard_summary_writer.add_scalar("train_loss", train_loss, epoch)
        tensorboard_summary_writer.add_scalar("train_accuracy", train_accuracy, epoch)
        
        # Test
        test_accuracy = test_model(model, test_set_loader)
        tensorboard_summary_writer.add_scalar("test_accuracy", test_accuracy, epoch)

        scheduler_dict = None
        if lr_scheduler:
            lr_scheduler.step()
            scheduler_dict = lr_scheduler.state_dict()

        # Save Checkpoint
        is_best = test_accuracy > best_test
        if is_best:
            best_test = test_accuracy
        if is_best or epochs < 15:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                logger.info("Saving checkpoint.")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_optimizer = accelerator.unwrap_model(optimizer)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': unwrapped_optimizer.state_dict(),
                # 'lr_scheduler_state_dict': scheduler_dict,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
                }, checkpoint_file_path)

        elapsed_time = t.stop()
        if accelerator.is_local_main_process:
            logger.info(f"End of epoch {epoch}, took {elapsed_time:0.4f} seconds.")
            logger.info(f"Average Train Loss: {train_loss}")
            logger.info(f"Top-1 Train Accuracy: {train_accuracy}")
            logger.info(f"Top-1 Test Accuracy: {test_accuracy}")
    if accelerator.is_local_main_process:
        logger.info(f"Top-1 Best Test Accuracy: {best_test}")

from functools import partial

if __name__ == '__main__':
    """
    Basic Example
    You will need to blow away the TensorBoard logs and checkpoint file if you want
    to train from scratch a second time.
    """
    marked_images_directory = "experiments/radioactive/marked_images"
    output_directory="experiments/radioactive/train_marked_classifier"
    tensorboard_log_directory="runs/train_marked_classifier"
    optimizer = lambda model : torch.optim.AdamW(model.parameters())
    epochs = 60
    dataloader_func = partial(get_data_loaders_cifar, marked_images_directory, False)
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)

    main(dataloader_func, model, optimizer, output_directory, tensorboard_log_directory,
         epochs=epochs)