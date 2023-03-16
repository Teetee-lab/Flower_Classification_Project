# Imports here
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
sns.set()
import torch
from torchvision import transforms, datasets, models
from torch.autograd import Variable
from torch import nn, optim
from collections import OrderedDict
import os
import copy
import argparse
import time
import torchvision.models as model
from PIL import Image
import argparse

#arguments to be parsed
parser = argparse.ArgumentParser(description = 'Flower Classification')
parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save checkpoint')
parser.add_argument('--gpu', type=bool, default=False, help='to use gpu')
parser.add_argument('--arch', type=str, default='VGG', help='architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help ='initial hidden units')
parser.add_argument('--epochs', type=int, default =5 , help='training epochs')

args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean =[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                         ])
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean =[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                     ])

val_transforms = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean =[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    ])                           

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(root = train_dir, transform =train_transforms)
test_datasets = datasets.ImageFolder(root = test_dir, transform =test_transforms)
val_datasets = datasets.ImageFolder(root = valid_dir, transform =val_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {w: torch.utils.data.DataLoader(z, batch_size = 4,shuffle = True, num_workers = 4) for w, z in [('train', train_datasets),('test', test_datasets),('val', val_datasets)]}

dataset_sizes = {w: len(z) for w,z in [('train', train_datasets), ('test', test_datasets),('val', val_datasets)]}
print(dataset_sizes)

# TODO: Build and train your network
#Pretrained network
model = models.vgg16_bn(pretrained =True)

#freeze parameters
for param in model.parameters():
    param.requires_grad = False
    
#print number of output
number_output = print(len(train_datasets.classes))
number_output

#transfer learning
number_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1] #remove the last layer
features.extend([torch.nn.Linear(number_features, 102)])
model.classifier = torch.nn.Sequential(*features)

#set criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), 
                       lr = args.learning_rate)

#check if using gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}.")
model = model.to(device)

#define function to train network
def train_model(model, criterion, optimizer, epochs=args.epochs):
    start = time.time()
    
    best_classifier_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
        
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs - 1}")
        print('-' * 10)
                       
        for each_phase in ['train', 'val']:
            if each_phase == 'train':
                model.train()
            else:
                model.eval()
                       
            step_loss = 0.0
            step_correct = 0
                       
            for inputs, labels in dataloaders[each_phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
            
                optimizer.zero_grad()
            
                with torch.set_grad_enabled(each_phase =='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if each_phase == 'train':
                        loss.backward()
                        optimizer.step()
                       
                       
                step_loss += loss.item() * inputs.size(0)
                step_correct += torch.sum(preds == labels.data)
        
        
            epoch_loss = step_loss /dataset_sizes[each_phase]
            epoch_accuracy = step_correct.double() / dataset_sizes[each_phase]
                       
            print(f"{each_phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}")
        
            if each_phase == 'val' and (epoch_accuracy > best_accuracy):
                best_accuracy = epoch_accuracy
                best_classifier_wts = copy.deepcopy(model.state_dict())
                       
        print()
                       
    stop = time.time() - start
    print(f"Training complete in {stop // 60:.0f}m, {stop % 60:.0f}s")
    print("-" * 10)
    print(f"Best val Acc: {best_accuracy:4f}")
                        
                       
    model.load_state_dict(best_classifier_wts)
    return model


#train the network
model = train_model(model, criterion, optimizer, epochs = args.epochs)


# TODO: Do validation on the test set
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in dataloaders['test']:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f"Accuracy of the network on the test images: {100 * correct // total} %")


# TODO: Save the checkpoint 
model.class_to_idx = train_datasets.class_to_idx
checkpoint = {'arch': args.arch,
             'input': number_features,
             'output': 102,
             'Epoch': args.epochs,
             'learning_rate': args.learning_rate,
             'dropout': 0.5,
             'batch_size': 4,
             'classifier': model.classifier,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'class_idx': model.class_to_idx}
torch.save(checkpoint, args.save_dir)