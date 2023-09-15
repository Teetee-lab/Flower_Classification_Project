# Flower_Classification_Project

![Flower Image](https://img.freepik.com/free-photo/purple-osteospermum-daisy-flower_1373-16.jpg)

## Overview
In this repository, I have implemented a method of transfer learning on a Vgg16_bn pre-trained model to train a dataset of flower categories. The output of the pretrained model is then used to build a command line application that take the image of a flower as input and predict its category.

## Dataset Overview
The dataset contains 102 categories of flowers, and the flowers are chosen to be the types of flowers commonly planted and grown in the United Kingdom. Each class of the flower consists of 40 and 258 images. Aside from that, the images have large scale, pose, and light variations. 

## Transformation
Due to the large scale of the images, they were randomly scaled, resized, and normalized in order to have similar records across the datasets.

## Results
The model performed at its best by predicted an accuracy of 82% on validation data and 90% on unseen flower images. 

## Next Step
I plan to increase the dataset by adding more flower images from other countries, also would like to incorporate other pretrained models.

## Installation for Command-line Application
 * Numpy
   -  conda install numpy
 * Matplotlib
   -  conda install matplotlib
 * Pil
   -  conda install -c anaconda pil
 * Pytorch
   -  conda install torch
 
## Instructions on how to run the Python Script
 - ### For the train.py the below argument needs to be passed;
     * data directory
       -  --data_dir
     * saved checkpoint directory
       -  --save_dir
     * GPU training
       -  --GPU
     * architecture
       -  --arch
     * learning rate
       -  --learning_rate
     * hidden units
       -  --hidden_units
     * Number of epochs
       -  --epochs
  
 - ### For the predict.py the below argument needs to be passed;
     * input image directory
       -  --image
     * checkpoint directory
       - --check_point
     * GPU usage
       -  --GPU
     * top k classes with probability
       -  --top_k
     * category to name
       -  --category_to_nam
  
## To run the Python file in the command line, use the below;
    * python train.py --data_dir dir_path --gpu True --learning_rate 0.001 after successful training.
   
    * Then run:
      python predict.py --image image_name.jpg --checkpoint checkpoint.pth --gpy True
   
## Repository Structure
   
      ├── Image Classifier Project notebook.ipynb     <- documentation of the project in Jupyter notebook            
      ├── train.py             <- training python file
      ├── predict.py           <- Prediction python file
      └── README.md            <- Top-level README
      
## Author

Titilayo Amuwo

  

 

