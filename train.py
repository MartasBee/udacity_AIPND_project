# +
# Imports

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt

import os
import json
import time
import argparse

from collections import OrderedDict
from PIL import Image

from flowers_common import generate_checkpoint_file_name, build_checkpoint_save_path, save_train_checkpoint, \
                            create_model_densenet121, create_model_resnet18


#----------------------------------------------------------------------------------------------------

def load_train_data(rootdir='../data/'):
    data_root = rootdir
    data_dir  = data_root + 'flowers'

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir  = data_dir + '/test'
    
    # DONE: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # DONE: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transforms)

    # DONE: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64)
    
    return train_dataloader, valid_dataloader, test_dataloader


#----------------------------------------------------------------------------------------------------

def train_model(model_dict, criterion, optimizer, device, \
                train_dataloader, valid_dataloader, \
                num_epochs=5, num_valid_steps=10, \
                checkpoints_save_dir=None):
    
    train_start_time = time.time()
    
    model_dict['model'].to(device)

    # setup learning params
    epochs = num_epochs
    validation_step = num_valid_steps
    steps = 0
    running_loss = 0    

    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        # store start time of current epoch
        epoch_start_time = time.time()
        
        tot_train_loss_in_epoch = 0
        steps = 0

        model_dict['model'].train()

        for inputs, labels in train_dataloader:
            steps += 1

            # move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # feed forward
            logps = model_dict['model'].forward(inputs)
            # calculate loss
            loss = criterion(logps, labels)

            # calculate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tot_train_loss_in_epoch += loss.item()

            # within current epoch, validate trainging progress after `validation_step` batches are processed
            if steps % validation_step == 0:
                valid_step_loss = 0
                accuracy = 0
                # switch model to evaluation mode
                model_dict['model'].eval()
                # with gradient calculation turned off
                with torch.no_grad():
                    for inputs, labels in valid_dataloader:
                        # move data to device
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model_dict['model'].forward(inputs)
                        valid_step_loss += criterion(logps, labels).item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print("Epoch {:>2}/{:>2}.. ".format(epoch+1, epochs),
                      "Processed images {:>4}.. ".format(steps*64),
                      "Train loss: {:.3f}.. ".format(running_loss/validation_step),
                      "Valid loss: {:.3f}.. ".format(valid_step_loss/len(valid_dataloader)),
                      "Valid accuracy: {:.3f}".format(accuracy/len(valid_dataloader)))

                running_loss = 0
                # switch model back to training mode
                model_dict['model'].train()

        # else end of epoch - all batches from train_dataloader are processed now
        else:
            tot_valid_loss_in_epoch = 0
            # Accuracy = Number of correct predictions on the valid set
            valid_correct = 0

            # set model to evaluation mode
            model_dict['model'].eval()
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for inputs, labels in valid_dataloader:
                    # move data to device
                    inputs, labels = inputs.to(device), labels.to(device)

                    logps = model_dict['model'].forward(inputs)
                    tot_valid_loss_in_epoch += criterion(logps, labels).item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    valid_correct += equals.sum().item()

            # set model back to train mode
            model_dict['model'].train()

            # Get mean loss to enable comparison between train and validation sets
            train_loss = tot_train_loss_in_epoch / len(train_dataloader.dataset)
            valid_loss = tot_valid_loss_in_epoch / len(valid_dataloader.dataset)

            # At completion of epoch
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            epoch_duration = time.time() - epoch_start_time
            
            print('-------------------------------------------------------------------------------')
            print("Epoch {:>2}/{:>2} FINISHED.. ".format(epoch+1, epochs),
                  "Train Loss: {:.3f}.. ".format(train_loss),
                  "Valid Loss: {:.3f}.. ".format(valid_loss),
                  "Valid Accuracy: {:.3f}".format(valid_correct / len(valid_dataloader.dataset)))
            print('Epoch completed in {:.0f}m {:.0f}s'.format(epoch_duration // 60, epoch_duration % 60))
            print('===============================================================================')
            
            if checkpoints_save_dir is not None:
                save_train_checkpoint(model_dict,
                                      epoch, 
                                      checkpoints_save_dir+generate_checkpoint_file_name(epoch=epoch))

    time_elapsed = time.time() - train_start_time
    
    if checkpoints_save_dir is not None:
        save_train_checkpoint(model_dict,
                              epoch, 
                              checkpoints_save_dir+generate_checkpoint_file_name())
    
    print('===============================================================================')
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('===============================================================================')

    
# DONE: Do validation on the test set
def test_trained_model(model_dict, criterion, device, test_dataloader):
    model_dict['model'].to(device)

    test_loss = 0
    accuracy = 0

    # switch model to evaluation mode
    model_dict['model'].eval()
    # with gradient calculation turned off
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            # move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model_dict['model'].forward(inputs)
            test_loss += criterion(logps, labels).item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print('===============================================================================')
    print("Test loss: {:.3f}.. ".format(test_loss/len(test_dataloader)),
          "Test accuracy: {:.3f}".format(accuracy/len(test_dataloader)))
    print('===============================================================================')


#----------------------------------------------------------------------------------------------------


def parse_arguments():
    # parse cmd line argument
    parser = argparse.ArgumentParser(prog='train',
                                     description='Train neural network to classify 102 species of flowers',)
    
    parser.add_argument('data_directory',
                        help='The path to directory with training data',
                        action='store', type=str)
    parser.add_argument('-a', '--arch',
                        help='Architecture: Densenet121 or Resnet18',
                        choices=['Densenet121', 'Resnet18'],
                        default='Resnet18',
                        action='store', type=str, required=False)
    parser.add_argument('-u', '--hidden_units',
                        help='Number of neurons in hidden layer',
                        default='256',
                        action='store', type=int, required=False)
    parser.add_argument('-l', '--learning_rate',
                        help='Learning rate',
                        default='0.005',
                        action='store', type=float, required=False)
    parser.add_argument('-e', '--epochs',
                        choices=range(1, 11),
                        default='5',
                        help='Number of epochs',
                        action='store', type=int, required=False)
    parser.add_argument('-g', '--gpu',
                        help='Train on GPU, if available',
                        action='store_true', required=False)    
    parser.add_argument('-s', '--save_dir',
                        help='Path to dir to store checkpoints',
                        default='../data/_trained_models/',
                        action='store', type=str, required=False)
    parser.add_argument('-v', '--verbose',
                        help='Show extra logs',
                        action='store_true', required=False)
    
    args = parser.parse_args()
    return args


def show_arguments(parsed_args):
    print("==== ==== ==== ==== ====")
    print('data_directory ', parsed_args.data_directory)
    print('arch           ', parsed_args.arch)
    print('hidden_units   ', parsed_args.hidden_units)
    print('learning_rate  ', parsed_args.learning_rate)
    print('epochs         ', parsed_args.epochs)
    print('gpu            ', parsed_args.gpu)
    print('save_dir       ', parsed_args.save_dir)

    
#----------------------------------------------------------------------------------------------------

    
def main():
    args = parse_arguments()
    if args.verbose:
        show_arguments(args)
    
    # load parsed arguments
    data_directory = args.data_directory
    save_dir = args.save_dir
    
    arch = args.arch
    hidden_units = args.hidden_units
    
    learning_rate = args.learning_rate
    epochs = args.epochs
    
    
    # check if provided data directory exists
    if not os.path.isdir(data_directory):
        print('ERROR: The \'data_directory\' specified ({}) does not exist'.format(data_directory))
        exit()
        
    # check if provided directory for storing checkpoints exists, if no, create it
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    
    # load datasets
    train_loader, valid_loader, test_loader = load_train_data(data_directory)
    
    
    # create model & appropriate optimizer
    if arch == 'Densenet121':
        model_dict = create_model_densenet121(hidden=hidden_units)
        optimizer  = optim.Adam(model_dict['model'].classifier.parameters(), lr=learning_rate)
    elif arch == 'Resnet18':
        model_dict = create_model_resnet18(hidden=hidden_units)
        optimizer  = optim.Adam(model_dict['model'].fc.parameters(), lr=learning_rate)
    else:
        print("ERROR: Unknown model requested. Exiting...")
        exit()
    
    criterion = nn.NLLLoss()
    
    if args.verbose:
        print("==== ==== ==== ==== ====")
        print('NN MODEL:')
        print(model_dict['model'])
        
        
    # Use GPU if available and if allowed by paramter "GPU"
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    if args.verbose:
        print("==== ==== ==== ==== ====")
        print('Device selected for training: ', device)
        
    
    # finally run training
    train_model(model_dict, criterion, optimizer, device, 
                train_loader, valid_loader,
                num_epochs=epochs,
                checkpoints_save_dir=build_checkpoint_save_path(save_dir))
        
    
    # test trained model
    test_trained_model(model_dict, criterion, device, test_loader)
    
    
    if args.verbose:
        print("==== ==== ==== ==== ====")
        print("TRAINING FINISHED")
        print("==== ==== ==== ==== ====")
    
        
if __name__ == "__main__":
    main()

