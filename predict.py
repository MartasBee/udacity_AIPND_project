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

from flowers_common import load_checkpoint, recreate_model_from_checkpoint, \
                           create_model_densenet121, create_model_resnet18, \
                           load_category_names, class_to_name_lookup, pred_class_idx_to_flower_name


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

def process_image_via_numpy(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # DONE: Process a PIL image for use in a PyTorch model
    
    # open PIL image
    with Image.open(image) as im:        
        # reize and crop
        pil_image = im.resize((256, 256))
        pil_image = pil_image.crop((16, 16, 240, 240))
        
        # transform to numpy with float values 0.0-1.0
        np_image = np.array(pil_image)
        np_image = np_image / 255    
    
    # normalization
    mean  = np.array([0.485, 0.456, 0.406])
    std   = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # transpone - color channel to be 1st dimention of the array
    np_image = np_image.transpose((2,0,1))
    
    return np_image


def process_image_via_torch(image):
    
    with Image.open(image) as im:
        pil_to_tensor = transforms.ToTensor()(im).unsqueeze_(0)
        
    inference_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    pil_to_tensor = inference_transforms(pil_to_tensor)
    return pil_to_tensor


#----------------------------------------------------------------------------------------------------


# display numpy
def imshow_numpy(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# credits: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# display tensor
# credits: https://pytorch.org/vision/stable/auto_examples/plot_scripted_tensor_transforms.html
def imshow_tensor(image_tensor):
    fix, axs = plt.subplots(ncols=len(image_tensor), squeeze=False)
    for i, img in enumerate(image_tensor):
        input_tensor = inverse_normalize(tensor=img,
                                         mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
        input_tensor = transforms.ToPILImage()(input_tensor.to('cpu'))
        
        axs[0, i].imshow(np.asarray(input_tensor))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    
def show_probablities_chart(probabilities, classes):
    
    # get list of labels
    labels = class_to_name_lookup(cat_to_name, classes)
    
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(classes)), probabilities)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(labels)
    
    plt.show()
    

#----------------------------------------------------------------------------------------------------

def predict(image_path_or_tensor, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # DONE: Implement the code to predict the class from an image file
    
    if type(image_path_or_tensor) == torch.Tensor:
        input_data = image_tensor
    else:
        input_data = process_image_via_torch(image_path_or_tensor)
    
    # move model to target device
    model.to(device)
    
    # switch model to evaluation mode
    model.eval()
    # with gradient calculation turned off
    with torch.no_grad():

        if type(input_data) != torch.Tensor:
            print("processing non-tensor array to tensor")
            input_data = torch.from_numpy(input_data)
            input_data.unsqueeze_(0)
            input_data = input_data.float()
        
        # move data to device
        input_data = input_data.to(device)
                
        logps = model.forward(input_data)
        
        # get top topk classes
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
    
    # move tensors to cpu to transform them into numpy and list
    top_p, top_class = top_p.cpu(), top_class.cpu()
    
    # torch.Tensor to numpy of float
    top_p = top_p.data.numpy().squeeze(axis=0)
    
    # torch.Tensor to list of ints
    top_class = top_class.squeeze().tolist()
    if type(top_class) != list:
        top_class = [top_class]
    
    return top_p, top_class


#----------------------------------------------------------------------------------------------------


def parse_arguments():
    # parse cmd line argument
    parser = argparse.ArgumentParser(prog='predict',
                                     description='Predict species of given flower',)
    
    parser.add_argument('image_to_predict',
                        help='The path to image for prediction',
                        action='store', type=str)
    
    parser.add_argument('model_checkpoint',
                        help='The path to trained network checkpoint',
                        action='store', type=str)
    
    parser.add_argument('-t', '--top_k',
                        choices=range(1, 11),
                        default='1',
                        help='Return top K most likely classes',
                        action='store', type=int, required=False)
    
    parser.add_argument('-c', '--category_names',
                        help='Path to category-to-name file',
                        action='store', type=str, required=False)
    
    parser.add_argument('-g', '--gpu',
                        help='Inference on GPU, if available',
                        action='store_true', required=False)    
    
    parser.add_argument('-v', '--verbose',
                        help='Show extra logs',
                        action='store_true', required=False)
    
    args = parser.parse_args()
    return args


def show_arguments(parsed_args):
    print("==== ==== ==== ==== ====")
    print('image_to_predict ', parsed_args.image_to_predict)
    print('model_checkpoint ', parsed_args.model_checkpoint)
    print('top_k            ', parsed_args.top_k)
    print('category_names   ', parsed_args.category_names)
    print('gpu              ', parsed_args.gpu)
    
    
#----------------------------------------------------------------------------------------------------


def main():
    args = parse_arguments()
    if args.verbose:
        show_arguments(args)
        
    # load parsed arguments
    image_to_predict = args.image_to_predict
    model_checkpoint = args.model_checkpoint
    category_names   = args.category_names
    top_k = args.top_k

    
    # check if provided image exists
    if not os.path.exists(image_to_predict):
        print('ERROR: The \'image_to_predict\' specified ({}) does not exist'.format(image_to_predict))
        exit()
        
    # check if provided checkpoint exists
    if not os.path.exists(model_checkpoint):
        print('ERROR: The \'model_checkpoint\' specified ({}) does not exist'.format(model_checkpoint))
        exit()
        
    cat_to_name_dict = {}
    # check if provided category file exists
    if not category_names:
        print('INFO: The \'category_names\' NOT specified.')
    elif not os.path.exists(category_names):
        print('WARNING: The \'category_names\' specified ({}) does not exist'.format(category_names))
    else:
        cat_to_name_dict = load_category_names(category_names)

        
    # load checkpoint and build the model
    loaded_checkpoint = load_checkpoint(model_checkpoint)
    
    loaded_model_dict = recreate_model_from_checkpoint(loaded_checkpoint)
    
    if args.verbose:
        print("==== ==== ==== ==== ====")
        print('NN MODEL:')
        print(loaded_model_dict['model'])
    
    
    # Use GPU if available and if allowed by paramter "GPU"
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    if args.verbose:
        print("==== ==== ==== ==== ====")
        print('Device selected for training: ', device)
        
        
    # predict class idx of given flower image
    predP, predCI = predict(image_to_predict, loaded_model_dict['model'], device, topk=top_k)
    predCL, predN = pred_class_idx_to_flower_name(loaded_checkpoint['class_idx_to_class'], cat_to_name_dict, predCI)
    
   
    print("==== ==== ==== ==== ==== ==== ==== ====")
    print("Input image: ", image_to_predict)
    if not predN:
        for cl, pr in zip(predCL, predP):
            print("Prediction:  class {:>3}   probability: {:.3f}".format(cl, pr))
    else:
        for cl, pr, la in zip(predCL, predP, predN):
            print("Prediction:  class {:>3}   probability: {:.3f}   label: {}".format(cl, pr, la))
    print("==== ==== ==== ==== ==== ==== ==== ====")
    
        
if __name__ == "__main__":
    main()

