# Smiles-Detector

## Summary

Using LeNet CNN architecture and OpenCV to detect Smiles in real-time using your own webcam. Just run the Jupyter notebook associated with this repository. Recieved lots of help implementing this project through the book "Deep Learning for Computer Vision with Python" by Dr. Adrian Rosebrock.

Below is what it looks like:

![alt text](https://github.com/JamesPNacino/Smiles-Detector/blob/master/Not%20Smiling.PNG)
![alt text](https://github.com/JamesPNacino/Smiles-Detector/blob/master/Smiling.PNG)

## Python Packages 

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

You will also need to have software installed to run and execute a Jupyter Notebook

If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included. This project was also immplemented using Python 3.

## Code
To implement this code and train the models in this project you would have to follow the step by step instructions in the file 'Smiles Project.ipynb'.

## Run
Setting up CUDA GPU computing with Tensorflow
In this following video are the steps I took to use get my GPU to start training neural networks. https://www.youtube.com/watch?v=io6Ajf5XkaM

I have used a NVIDIA GTX 1070 GPU for this project to speed up computing times.

## Data Required for this project
https://github.com/opencv/opencv
https://github.com/hromi/SMILEsmileD

Through this link you will be able to find the Haar cascade filter used to detect faces. You will also need to download the pictures throught the second link - the pictures to train on are not uploaded to this repo.
