#IMPORTS
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

# using argparse library to parse arguments
parser = argparse.ArgumentParser(description = "Prediction script")
parser.add_argument('image_dir', help = 'Path of images. Mandatory', type = str)
parser.add_argument('load_dir', help = 'Path of checkpoint. Mandatory', type = str)
parser.add_argument('--top_k', help = 'Top K most likely classes. Optional', type = int)
parser.add_argument('--category_names', help = 'Mapping of categories to labels JSON. Optional', type = str)
parser.add_argument('--GPU', help = "Option to use GPU. Optional", type = str)

# Function to load a NN model
def load_model(file_path):
    checkpoint = torch.load(file_path)
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained = True)
    # VGG13
    else: 
        model = models.vgg13(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
    # Turn off Gradient
    for param in model.parameters():
        param.requires_grad = False

    return model

# Same function used in Notebook
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = Image.open(image)
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transformations(img)
    return img
#Same function from Notebook
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to("cuda")
    i = process_image(image_path)
    i = i.unsqueeze_(0)
    i = i.float()
    
    with torch.no_grad():
        output = model.forward(i.cuda())
    probability = F.softmax(output.data,dim=1)
    return probability.topk(topk)

#setting values data loading
args = parser.parse_args()
file_path = args.image_dir

#defining device: either cuda or cpu
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

#loading JSON file if provided, else load default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

#loading model from checkpoint provided
model = load_model(args.load_dir)

#defining number of classes to be predicted. Default = 1
if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1

#calculating probabilities and classes
probabilities = predict(file_path, model, nm_cl)

#preparing class_names using mapping with cat_to_name
class_names = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])
for l in range(nm_cl):
    print("Number: {}/{}.. ".format(l+1, nm_cl),
    "Class name: {}.. ".format(class_names[l]),
    "Probability: {}".format(probability[l]*100),
    )
    
    
## python3 predict.py 'flowers/test/10/image_07104.jpg' checkpoint.pth