# Flower_Classification_Project

![Flower Image](https://img.freepik.com/free-photo/purple-osteospermum-daisy-flower_1373-16.jpg)

## Overview
In this repository, I have implemented a command line application to train a model and then using the trained model you can input a flower image to predict a flower category. I have used method of transfer learning on a Vgg16_bn pretrained model.

## Dataset Overview
There are 102 category dataset, consisting of flower categories. The flowers chosen to be flower commonly occuring in the United Kingdom. Each class consists of between 40 and 258 images. The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The images were randomly scaled, resized also normalized in order to have similar records across the datasets.

## Results
The model predicted an accuracy of 82% on validation data and 90% on unseen flower image. 

## Next Step
In the future, I will like to increase the data and also try other pretrained models for more improved accuracy.

## Installation for Command line Application
 * Numpy
   -  conda install numpy
 * Matplotlib
   -  conda install matplotlib
 * Pil
   -  conda install -c anaconda pil
 * Pytorch
   -  conda install torch
 
## Instructions on how to run the Python Script
 - ### For the train.py the below argument need to be passed;
     * data directory
       -  --data_dir
     * saved checkpoint directory
       -  --save_dir
     * gpu training
       -  --gpu
     * architecture
       -  --arch
     * learning rate
       -  --learning_rate
     * hidden units
       -  --hidden_units
     * number of epochs
       -  --epochs
  
 - ### For the predict.py the below argument need to be passed;
     * input image directory
       -  --image
     * checkpoint directory
       - --check_point
     * gpu usage
       -  --gpu
     * top k classes with probability
       -  --top_k
     * category to name
       -  --category_to_nam
  
## To run the python file in the command line, use the below;
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

  

 

