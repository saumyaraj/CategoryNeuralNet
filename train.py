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

parser = argparse.ArgumentParser(description = "Neural Network training script")

parser.add_argument('data_dir', help = 'Directory to look for data. Mandatory', type = str)
parser.add_argument('--save_dir', help = 'Directory where model will be saved. Optional', type = str)
parser.add_argument('--arch', help = 'Default Alexnet will be used but if this arg specfied than VGG13 can be used, ', type = str)
parser.add_argument('--lrn', help = 'Learning rate, Default value = 0.001', type = float)
parser.add_argument('--hidden_units', help = 'Hidden units. Default value = 2048', type = int)
parser.add_argument('--epochs', help = 'Epochs. Default = 5', type = int)
parser.add_argument('--GPU', help = "GPU option", type = str)

args = parser.parse_args()
data_dir = args.data_dir
training_dir = data_dir + '/train/'
validation_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#defining device: either cuda or cpu
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'


training_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])

validation_data_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])

test_data_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])
# Loading datasets with ImageFolder
training_image_datasets = datasets.ImageFolder(training_dir, transform = training_data_transforms)
validation_image_datasets = datasets.ImageFolder(validation_dir, transform = validation_data_transforms)
test_image_datasets = datasets.ImageFolder(test_dir, transform = test_data_transforms)

# Dataloaders
training_loader = torch.utils.data.DataLoader(training_image_datasets, batch_size = 64, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_image_datasets, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size = 64, shuffle = True)

#mapping from category label to category name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model(arch, hidden_units):
    if arch == 'vgg13':
        model = models.vgg13(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential (OrderedDict([
                           ('fc1', nn.Linear(25088, 4096)),
                           ('relu1', nn.ReLU()),
                           ('dropout1', nn.Dropout(p = 0.3)),
                           ('fc2', nn.Linear(4096, hidden_units)),
                           ('relu2', nn.ReLU()),
                           ('dropout2', nn.Dropout(p = 0.3)),
                           ('fc3', nn.Linear(hidden_units, 102)),
                           ('output', nn.LogSoftmax(dim =1))
                            ]))
        else: #if hidden_units not given
            classifier = nn.Sequential (OrderedDict([
                       ('fc1', nn.Linear(25088, 4096)),
                       ('relu1', nn.ReLU()),
                       ('dropout1', nn.Dropout(p = 0.3)),
                       ('fc2', nn.Linear(4096, 2048)),
                       ('relu2', nn.ReLU()),
                       ('dropout2', nn.Dropout(p = 0.3)),
                       ('fc3', nn.Linear(2048, 102)),
                       ('output', nn.LogSoftmax(dim =1))
                        ]))
    else: #default alexnet model
        arch = 'alexnet' #will be used for checkpoint saving, so should be explicitly defined
        model = models.alexnet(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            classifier = nn.Sequential (OrderedDict([
                           ('fc1', nn.Linear(9216, 4096)),
                           ('relu1', nn.ReLU()),
                           ('dropout1', nn.Dropout(p = 0.3)),
                           ('fc2', nn.Linear(4096, hidden_units)),
                           ('relu2', nn.ReLU()),
                           ('dropout2', nn.Dropout(p = 0.3)),
                           ('fc3', nn.Linear(hidden_units, 102)),
                           ('output', nn.LogSoftmax(dim =1))
                            ]))
        else:
            classifier = nn.Sequential (OrderedDict([
                       ('fc1', nn.Linear(9216, 4096)),
                       ('relu1', nn.ReLU()),
                       ('dropout1', nn.Dropout(p = 0.3)),
                       ('fc2', nn.Linear(4096, 2048)),
                       ('relu2', nn.ReLU()),
                       ('dropout2', nn.Dropout(p = 0.3)),
                       ('fc3', nn.Linear(2048, 102)),
                       ('output', nn.LogSoftmax(dim =1))
                        ]))
    model.classifier = classifier
    return model, arch

def validation(model, valid_loader, criterion):
    model.to(device)

    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality =(labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

#loading model using above defined functiion
model, arch = load_model(args.arch, args.hidden_units)

#Actual training of the model
#initializing criterion and optimizer
criterion = nn.NLLLoss()

# get learning rate from parameter if provided or use default 0.001
if args.lrn:
    lr = args.lrn
else:
    lr = 0.001
optimizer = optim.Adam(model.classifier.parameters(), lr)


model.to(device)
if args.epochs:
    epochs = args.epochs
else:
    epochs = 5

print_every = 40
steps = 0

#runing through epochs
for i in range(epochs):
    running_loss = 0
    for(inputs, labels) in training_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() #where optimizer is working on classifier paramters only

        # Forward and backward passes
        outputs = model.forward(inputs) #calculating output
        loss = criterion(outputs, labels) #calculating loss(cost function)
        loss.backward()
        optimizer.step() #performs single optimization step
        running_loss += loss.item() # loss.item() returns scalar value of Loss function

        if steps % print_every == 0:
            model.eval() #switching to evaluation mode so that dropout is turned off
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validation_loader, criterion)

            print("Epoch: {}/{}.. ".format(i+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validation_loader)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(validation_loader)*100))

            running_loss = 0
            # Make sure training is back on
            model.train()

#saving trained Model
model.to('cpu') #no need to use cuda for saving/loading model.
# Save the checkpoint
model.class_to_idx = training_image_datasets.class_to_idx #saving mapping between predicted class and class name,
#second variable is a class name in numeric

#creating dictionary for model saving
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'arch': arch,
              'mapping':    model.class_to_idx
             }
#saving trained model for future use
if args.save_dir:
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save(checkpoint, 'checkpoint.pth')
    
## python3 train.py flowers