# +
# Imports

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

import numpy as np

import os
import json

from collections import OrderedDict


#----------------------------------------------------------------------------------------------------
# GENERATE PATH, SAVE and LOAD CHECKPOINT
#----------------------------------------------------------------------------------------------------

# DONE: Write a function that loads a checkpoint and rebuilds the model

def generate_checkpoint_file_name(filename='checkpoint.pth', epoch=None):
    # add epoch name to filename if required
    if epoch is not None:
        filename = 'epoch_' + str(epoch) + '_' + filename
        
    return filename
    
    
def build_checkpoint_save_path(root_data_dir, store_in_pwd=False):
    # if user wanna store data in current PWD
    if store_in_pwd:
        checkpoint_root = './'
    elif not root_data_dir:
        return None
    else:
        checkpoint_root = root_data_dir
    # checkpoint's dir
    data_dir = checkpoint_root + 'classifier_flowers/'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # and finally return complete path
    return data_dir


#----------------------------------------------------------------------------------------------------
# for saving model after each epoch
def save_train_checkpoint(model_dict, optimizer, epoch, idx_to_class, checkpoint_path):
   
    checkpoint = {'model_name': model_dict['name'],
                  'model_state_dict': model_dict['model'].state_dict(),
                  'classifier_hidden': model_dict['hidden'],
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': epoch,
                  'class_idx_to_class': idx_to_class}
    
    torch.save(checkpoint, checkpoint_path)


#----------------------------------------------------------------------------------------------------
# load state dict of the model
def load_checkpoint(checkpoint_path):
    # load required checkpoint
    checkpoint = torch.load(checkpoint_path)
        
    return checkpoint


def recreate_model_from_checkpoint(checkpoint):
    # based on model name, stored in checkpoint, recreate the model
    # Densenet121 based model
    if checkpoint['model_name'] == 'flowers_classifier_model__densenet':
        loaded_model_dict = create_model_densenet121(checkpoint['classifier_hidden'])
    # Resnet18 based model
    elif checkpoint['model_name'] == 'flowers_classifier_model__resnet':
        loaded_model_dict = create_model_resnet18(checkpoint['classifier_hidden'])
    else:
        print("ERROR: Unknown model name loaded from given checkpoint...")
        exit()
    
    loaded_model_dict['model'].load_state_dict(checkpoint['model_state_dict'])
    
    return loaded_model_dict


#----------------------------------------------------------------------------------------------------
# CREATE MODEL
#----------------------------------------------------------------------------------------------------

def create_model_densenet121(hidden=256):
    # load pretrained model from torchvision
    model = models.densenet121(pretrained=True)
    
    # create new classifier
    classifier = nn.Sequential(OrderedDict([
        ('full_1', nn.Linear(1024, hidden)),
        ('ReLU_1', nn.ReLU()),
        ('drop_1', nn.Dropout(0.2)),
        ('full_3', nn.Linear(hidden, 102)),
        ('soft', nn.LogSoftmax(dim=1))
    ]))
        
    # freeze model's parameters
    for param in model.parameters():
        param.requires_grad = False

    # exchange pretrained classifier in model with this custom one
    model.classifier = classifier
    
    return {'name':   'flowers_classifier_model__densenet',
            'model':  model,
            'hidden': hidden}


#----------------------------------------------------------------------------------------------------

def create_model_resnet18(hidden=256):
    # load pretrained model from torchvision
    model = models.resnet18(pretrained=True)
    
    # create new classifier
    classifier = nn.Sequential(OrderedDict([
        ('full_1', nn.Linear(512, hidden)),
        ('ReLU_1', nn.ReLU()),
        ('drop_1', nn.Dropout(0.2)),
        ('full_3', nn.Linear(hidden, 102)),
        ('soft', nn.LogSoftmax(dim=1))
    ]))
        
    # freeze model's parameters
    for param in model.parameters():
        param.requires_grad = False

    # exchange pretrained classifier in model with this custom one
    model.fc = classifier
    
    return {'name':   'flowers_classifier_model__resnet',
            'model':  model,
            'hidden': hidden}


#----------------------------------------------------------------------------------------------------
# LOAD CATEGORY NAMES FILE
#----------------------------------------------------------------------------------------------------

def load_category_names(cat_name_file):
    with open(cat_name_file, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name


def class_to_name_lookup(class_to_names_dict, predicted_classes_list):
    '''
    @param class_names_dict dictionary which maps class numbers to names
    @param predicted_classes_list list of classes (numbers with type str)
    @return list of labels
    '''
    labels = []
    for c in predicted_classes_list:
        labels.append(class_to_names_dict.get(str(c)))
        
    return labels


def pred_class_idx_to_flower_name(dataset_class_idx_to_class_list, class_to_names_dict, predicted_class_idx_list):
    predCL = []
    for ci in predicted_class_idx_list:
        predCL.append(dataset_class_idx_to_class_list[ci])
    predN = class_to_name_lookup(class_to_names_dict, predCL)
    
    return predCL, predN


