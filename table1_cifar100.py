import os
import glob
import shutil
from functools import partial

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cutie
import matplotlib.pyplot as plt

import resnet18.resnet18_cifar10 as resnet18cifar10
from models.resnet2 import resnet 
from radioactive.make_data_radioactive import get_images_for_marking_cifar10, get_images_for_marking_multiclass_cifar10
from radioactive.make_data_radioactive import main as do_marking
from radioactive.train_marked_classifier import main as train_marked_classifier
from radioactive.train_marked_classifier import get_data_loaders_cifar
from radioactive.detect_radioactivity import main as detect_radioactivity
from utils.utils import NORMALIZE_CIFAR10

import logging
from utils.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

"""
1. Train a resnet18 classifier on cifar10
2. Generate 1, 2, 5, 10% Markings using network from step 1.
3. Retrain the logistic regression layer on a copy of the trained resnet18 from step 1 using the marked
   images merged with vanilla images.
4. Perform the radioactive detection p-tests on the network trained in step 3. Compare the top-1 accuracy
   of this network with the network trained in step 1.
5. Generate Table 1.

NOTE: Augmentations aren't used in steps 2,3.
"""

def do_marking_run(class_marking_percentage, experiment_directory, tensorboard_log_directory, augment=True):

    # Setup experiment directory
    if os.path.isdir(experiment_directory):
        error_message = f"Directory {experiment_directory} already exists. By default we assume you don't want to "\
                        "repeat the marking stage."
        logger.info(error_message)
        return

    logfile_path = os.path.join(experiment_directory, 'marking.log')
    setup_logger_tqdm(filepath=logfile_path)

    # Load randomly sampled images from random class along with list of original indexes 
    training_set = torchvision.datasets.CIFAR100(root="experiments/datasets", download=True)
    class_id, images, original_indexes = get_images_for_marking_cifar10(training_set, 
        class_marking_percentage=class_marking_percentage,
        tensorboard_log_directory=tensorboard_log_directory)

    # Marking network is the resnet18 we trained on CIFAR10
    # marking_network = torchvision.models.resnet18(pretrained=False, num_classes=10)
    marking_network = resnet(num_classes=100, depth=164, block_name='bottleneck')
    checkpoint_path = "experiments/cifar100/table1/step1/checkpoint.pth"
    marking_network_checkpoint = torch.load(checkpoint_path)
    marking_network.load_state_dict({k.replace("module.", ""): v for k, v in 
        marking_network_checkpoint["model_state_dict"].items()})

    # Carriers
    # marking_network_fc_feature_size = 512
    marking_network_fc_feature_size = 256
    carriers = torch.randn(len(training_set.classes), marking_network_fc_feature_size)
    carriers /= torch.norm(carriers, dim=1, keepdim=True)
    torch.save(carriers, os.path.join(experiment_directory, "carriers.pth"))

    # Run!
    optimizer = lambda x : torch.optim.AdamW(x, lr=0.1)
    epochs = 100
    batch_size = 32
    output_directory = os.path.join(experiment_directory, "marked_images") 
    if not augment:
        augmentation = None
    marked_images = do_marking(output_directory, marking_network, images, original_indexes, carriers, 
                               class_id, NORMALIZE_CIFAR10, optimizer, tensorboard_log_directory, epochs=epochs, 
                               batch_size=batch_size, overwrite=True, augmentation=augmentation)

    # Show marked images in Tensorboard
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)
    images_for_tensorboard = [transforms.ToTensor()(x) for x in marked_images]
    img_grid = torchvision.utils.make_grid(images_for_tensorboard, nrow=16)
    tensorboard_summary_writer.add_image('marked_images', img_grid)

    # Record marking completion
    with open(os.path.join(experiment_directory, "marking.complete"),"w") as fh:
        fh.write("1")

def do_marking_run_multiclass(overall_marking_percentage, experiment_directory, tensorboard_log_directory, augment=True):
    # Setup experiment directory
    if os.path.isdir(experiment_directory):
        error_message = f"Directory {experiment_directory} already exists. By default we assume you don't want to "\
                        "repeat the marking stage."
        logger.info(error_message)
        return

    os.makedirs(experiment_directory)

    logfile_path = os.path.join(experiment_directory, 'marking.log')
    setup_logger_tqdm(filepath=logfile_path)

    training_set = torchvision.datasets.CIFAR100(root="experiments/datasets", download=True)

    # Marking network is the resnet18 we trained on CIFAR10
    # marking_network = torchvision.models.resnet18(pretrained=False, num_classes=10)
    marking_network = resnet(num_classes=100, depth=164, block_name='bottleneck')
    checkpoint_path = "experiments/cifar100/table1/step1/checkpoint.pth"
    marking_network_checkpoint = torch.load(checkpoint_path)
    marking_network.load_state_dict({k.replace("module.", ""): v for k, v in 
        marking_network_checkpoint["model_state_dict"].items()})

    # Carriers
    # marking_network_fc_feature_size = 512
    marking_network_fc_feature_size = 256
    carriers = torch.randn(len(training_set.classes), marking_network_fc_feature_size)
    carriers /= torch.norm(carriers, dim=1, keepdim=True)
    torch.save(carriers, os.path.join(experiment_directory, "carriers.pth"))


    # { 0 : [(image1, original_index1),(image2, original_index2)...], 1 : [....] }
    image_data = get_images_for_marking_multiclass_cifar10(training_set,
                                                           tensorboard_log_directory,
                                                           overall_marking_percentage)

    marked_images = []
    for class_id, image_list in image_data.items():
        if image_list:
            images, original_indexes = map(list, zip(*image_list))
            optimizer = lambda x : torch.optim.AdamW(x, lr=0.1)
            epochs = 100
            batch_size = 32
            output_directory = os.path.join(experiment_directory, "marked_images")
            if not augment:
                augmentation = None

            tensorboard_class_log = os.path.join(tensorboard_log_directory, f"class_{class_id}")
            marked_images_temp = do_marking(output_directory, marking_network, images, original_indexes, carriers, 
                                            class_id, NORMALIZE_CIFAR10, optimizer, tensorboard_class_log, epochs=epochs, 
                                            batch_size=batch_size, overwrite=False, augmentation=augmentation)
            
            marked_images =  marked_images + marked_images_temp   

    # Show marked images in Tensorboard
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)
    images_for_tensorboard = [transforms.ToTensor()(x) for x in marked_images]
    img_grid = torchvision.utils.make_grid(images_for_tensorboard, nrow=16)
    tensorboard_summary_writer.add_image('marked_images', img_grid)

    # Record marking completion
    with open(os.path.join(experiment_directory, "marking.complete"),"w") as fh:
        fh.write("1")

def do_training_run(run_name, augment):
    # Load our trained resnet18 from step1
    # model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model = resnet(num_classes=100, depth=164, block_name='bottleneck')
    checkpoint_path = "experiments/cifar100/table1/step1/checkpoint.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict({k.replace("module.", ""): v for k, v in 
        checkpoint["model_state_dict"].items()})

    # Retrain the fully connected layer only
    for param in model.parameters():
        param.requires_grad = False
    num_classes = 100
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    optimizer = lambda model : torch.optim.AdamW(model.fc.parameters())

    tensorboard_log_directory = f"runs/table1_{run_name}_target"
    epochs = 20
    output_directory = f"experiments/cifar100/table1/{run_name}/marked_classifier"
    marked_images_directory = f"experiments/cifar100/table1/{run_name}/marked_images"
    dataloader_func = partial(get_data_loaders_cifar, marked_images_directory, augment)
    train_marked_classifier(dataloader_func, model, optimizer, output_directory, tensorboard_log_directory, 
                            epochs=epochs)

def calculate_p_values(marking_percentages):
    logfile_path = f"experiments/cifar100/table1/detect_radioactivity.log"
    setup_logger_tqdm(logfile_path)

    p_values = []

    for run in marking_percentages:
        run_name = f"{run}_percent"
        carrier_path = f"experiments/cifar100/table1/{run_name}/carriers.pth"

        # target_network = torchvision.models.resnet18(pretrained=False, num_classes=10)
        target_network = resnet(num_classes=100, depth=164, block_name='bottleneck')
        target_checkpoint_path = f"experiments/cifar100/table1/{run_name}/marked_classifier/checkpoint.pth"
        target_checkpoint = torch.load(target_checkpoint_path)
        target_network.load_state_dict({k.replace("module.", ""): v for k, v in 
            target_checkpoint["model_state_dict"].items()})
        target_network.fc = nn.Sequential()

        # No need to align when only retraining the logistic regression
        (scores, p_vals, combined_pval) = detect_radioactivity(carrier_path, None, 
                                                               None, target_checkpoint,
                                                               align=False)
        p_values.append(combined_pval)

    return p_values

def generate_table_1(marking_percentages, p_values):
    # Get Vanilla Accuracy
    vanilla_checkpoint_path = "experiments/cifar100/table1/step1/checkpoint.pth"
    vanilla_checkpoint = torch.load(vanilla_checkpoint_path)

    # The Rest
    accuracies = [vanilla_checkpoint["test_accuracy"]]
    for run in marking_percentages:
        run_name = f"{run}_percent"
        marked_checkpoint_path = f"experiments/cifar100/table1/{run_name}/marked_classifier/checkpoint.pth"
        marked_checkpoint = torch.load(marked_checkpoint_path)
        accuracies.append(marked_checkpoint["test_accuracy"])

    # Create the table!
    column_labels = tuple([0] + marking_percentages)
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(column_labels)))
    row_labels = ["log10(p)", "Top-1 %"]
    formatted_pvalues = ["n/a"]
    formatted_pvalues += [f"{p:0.4f}" for p in np.log10(p_values)]

    cell_text = [formatted_pvalues, accuracies]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axis('off')
    table = ax.table(cellText=cell_text,
                     rowLabels=row_labels,
                     colColours=colors,
                     colLabels=column_labels,
                     loc='center')
    plt.savefig("experiments/cifar100/table1/table1.png")
    plt.show()


if __name__ == '__main__':
    marking_percentages = [1, 2, 5, 10, 20]
    # marking_percentages = [10]

    p_values_file = "experiments/cifar100/table1/p_values.pth"

    # Step 1 - Train Marking Network
    optimizer = lambda x : torch.optim.AdamW(x)
    output_directory_root = "experiments/cifar100/table1"
    experiment_name = "step1"
    epochs = 60
    # resnet18cifar10.main(experiment_name, optimizer, 
    #                      output_directory_root=output_directory_root,
    #                      epochs=epochs)

    # Step 2 - Marking
    for marking_percentage in marking_percentages:
        experiment_directory = os.path.join("experiments/cifar100/table1", f"{marking_percentage}_percent")
        if os.path.exists(os.path.join(experiment_directory, "marking.complete")):
            message = f"Marking step already completed for {marking_percentage}%. Do you want to restart this part of " \
                      "the experiment?"
            if not cutie.prompt_yes_or_no(message, yes_text="Restart", no_text="Skip marking step"):
                continue

        tensorboard_log_directory = os.path.join("runs", "table1", f"{marking_percentage}_percent", "marking")
        shutil.rmtree(experiment_directory, ignore_errors=True)
        shutil.rmtree(tensorboard_log_directory, ignore_errors=True)
        do_marking_run_multiclass(marking_percentage, experiment_directory, tensorboard_log_directory, augment=False)

    # Step 3 - Training Target Networks
    for marking_percentage in marking_percentages:
        do_training_run(f"{marking_percentage}_percent", augment=True)

    # Step 4 - Calculate p-values
    p_values = calculate_p_values(marking_percentages)  
    torch.save(p_values, p_values_file)
    p_values = torch.load(p_values_file)

    # Step 5 - Generate Table 1
    generate_table_1(marking_percentages, p_values)
