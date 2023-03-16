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
import time
import torchvision.models as model
from PIL import Image
import argparse

#check if using gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}.")

parser = argparse.ArgumentParser(description = "Prediction of the flower class")
parser.add_argument('--image', type=str, default = 'flowers/test/35/image_06986.jpg', help = 'image to be classified')
parser.add_argument('--checkpoint', type=str, default= 'checkpoint.pth', help = 'load the checkpoint')
parser.add_argument('--gpu', type=bool, default=False, help='to use gpu')
parser.add_argument('--top_k', type=int, default=5, help='top K classes with probabilities')
parser.add_argument('--category_to_name', type=str, default='cat_to_name.json', help='json mapping of category of image to name')

args = parser.parse_args()

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    if checkpoint['arch'] == 'vgg16_bn':
        model = models.vgg16_bn(pretrained = True)
    else:
        None
    model.to(device)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_idx']
    return model

def process_image(image_dir):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_dir)
    width = image.size[0]
    height = image.size[1]
    area = width/height
    
    if width <= height:
        image = image.resize((256,int(256/area)))
    else:
        image = image.resize((int(256*area),256))
    
    mid_width = image.size[0]/2
    mid_height = image.size[1]/2
    
    crop = image.crop((mid_width-112, mid_height-112, mid_width+112, mid_height+112 ))
    
    np_image = np.asarray(crop)/225
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image =(np_image - mean)/std
    image = image.transpose((2,0,1))
    
    return torch.from_numpy(image)

def predict(image_dir, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_dir)
    image = image.unsqueeze(0).float()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        'cpu'
        
    image = image.to(device)
    
    model = load_checkpoint(model)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
    
    probs, index = torch.topk(ps, top_k)
    p_b = np.array(probs.data[0])
    Index = np.array(index.data[0])
    
    with open(args.category_to_name, 'r') as f:
        cat_to_name = json.load(f)
    
    idx_to_class = {idx:clas for clas,idx in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in Index]
    labels = [cat_to_name[clas] for clas in classes]
    
    return p_b,labels

#Print the top K classes along with it's probabilities
probability, classes = predict(args.image, args.checkpoint, args.top_k)
print('Left: Possible Type Right: Probability')
for p_b, clas in zip(probability, classes):
    print("%20s: %f" % (clas, p_b))







