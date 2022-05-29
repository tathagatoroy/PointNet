#importing the neccesary libraries


import numpy as np
#import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
#import cv2
import matplotlib.pyplot as plt
import os 
import plotly.graph_objects as go
import plotly.express as px
import time
import random
from torch.utils.data import Dataset
from torch_geometric.datasets import ModelNet
import torch
import math
from torch.utils.data import DataLoader



#now writing the actual deeplearning model
# the input will be an 1024 * 3 Matrix describing a pointcloud and the output would be an 10 dim vector containing the class probabilities for classification
# this performs the transformation part of the network
# takes input and output dimension as input
# this is a generic network class so it may not have a specific input dimension
# hence it is given as the input dimension

class TNet(nn.Module):
  def __init__(self, input_dim , output_dim):
    super(TNet, self).__init__()

    #now break it into parts
    # input is torch object
    # dimension (BATCH_SIZE , NUM_POINTS, 3)

    self.input_dim = input_dim # value is 3
    self.output_dim = output_dim


    # input is of the form (B,3,N) = (32,3,1024)
    # 
    self.transformsConvolution = nn.Sequential(
        nn.Conv1d(self.input_dim,64,1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        #output is (32,64,1024)
        nn.Conv1d(64,128,1),
        nn.BatchNorm1d(128),
        nn.ReLU(),

        #output is (32,128,1024)

        nn.Conv1d(128,1024,1),
        nn.BatchNorm1d(1024),
        nn.ReLU(),

        #output is (32,1024,1024)

        nn.MaxPool1d(1024) # equivalent to number of sampled point

        #output is (32,1024,1)








        

    ) 

    #input should be 32 * 1024
    self.transformsFullyConnected = nn.Sequential(
         nn.Linear(1024,512),
         nn.BatchNorm1d(512),
         nn.ReLU(),

         #output is (32,512)

         nn.Linear(512,256),
         nn.BatchNorm1d(256),
         nn.ReLU(),

         nn.Linear(256,self.output_dim * self.output_dim)
         




    )

  def forward(self,x):

    #perform the convolution operations
    for index,layer in enumerate(self.transformsConvolution):
      
      x = layer(x)

    
    #outputs in (BATCH_SIZE,1024,1024)
    x = x.view(-1, 1024)
    for index,layer in enumerate(self.transformsFullyConnected):
      x = layer(x)
    #should output a vector of size (BATCH,output_dim * output_dim)

    #Adding the identity matrix to the output 
    I = torch.eye(self.output_dim)
    if torch.cuda.is_available():
        I = I.cuda()
    #print(x.view(-1,self.output_dim,self.output_dim).shape)
    x = x.view(-1, self.output_dim, self.output_dim) + I
    return x


    



# this is the base point net structure which uses two Tnet along with point encoding and local feature aggregration.
# would have to run a run through to figure out the exact dimension changes
# 1D convolution with kernel size = 1 is equivalent to a Shared MLP operation
class BasePointNet(nn.Module):

  def __init__(self, pointDimension , returnLocalFeatures = False):
    super(BasePointNet, self).__init__()


    self.returnLocalFeatures = returnLocalFeatures
    self.inputTransform = TNet(input_dim = pointDimension , output_dim = pointDimension)
    self.featureTransform = TNet(input_dim = 64, output_dim = 64)

    self.Convolution1 = nn.Sequential(
        nn.Conv1d(3,64,1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        
        nn.Conv1d(64,64,1),
        nn.BatchNorm1d(64),
        nn.ReLU()


    )

    self.Convolution2 = nn.Sequential(
        

        nn.Conv1d(64,64,1),
        nn.BatchNorm1d(64),
        nn.ReLU(),

        nn.Conv1d(64,128,1),
        nn.BatchNorm1d(128),
        nn.ReLU(),


        nn.Conv1d(128,1024,1),
        nn.BatchNorm1d(1024),
        nn.ReLU()



  )


  def forward(self,x):
    #print("Initial shape of X : {0} ".format(x.shape))
    numPoints = x.shape[2]

    InputTransform = self.inputTransform(x)
    #print("Shape of the Input transform : {0}".format(InputTransform.shape))

    
    #batch matrix multiplication
    x = x.transpose(2,1)
    #print("Shape of x is : {0}".format(x.shape))

    x = torch.bmm(x, InputTransform)
    #print("Shape after the batch matrix multiplication : {0} ".format(x.shape))
    x = x.transpose(2,1)
    #print("Shape after transpose operation : {0}".format(x.shape))

    #first convolution

    for index,layer in enumerate(self.Convolution1):
      x = layer(x)

    # shapen after First Convolution 
    #print("Shape after first set of Convolution Operation is {0}".format(x.shape))


    FeatureTransform = self.featureTransform(x)


    x = x.transpose(2,1)
    #print("Shape after transpose : {0}".format(x.shape))

    
    #print("Shape of the Feature Transform : {0}".format(FeatureTransform.shape))

    #local Features
    x = torch.bmm(x,FeatureTransform)

    localPointFeatures = x 

    #print("Shape after Feature Transform : {0}".format(x.shape))

    x = x.transpose(2,1)
    #print("After Transpose Operation, the shape is {0}".format(x.shape))


    for index,layer in enumerate(self.Convolution2):
      x = layer(x)

    #print("Shape after Convolution 2 is {0}".format(x.shape))

    # Max Pooling operation
    x = nn.MaxPool1d(numPoints)(x)
    #print(x.shape)
    x = x.view(-1, 1024)

    #print("shape after max pool operation : {0}".format(x.shape))


    if self.returnLocalFeatures:
            x = x.view(-1, 1024, 1).repeat(1, 1, numPoints)
            return torch.cat([x.transpose(2, 1), localPointFeatures], 2), FeatureTransform
    else:
            return x, FeatureTransform


#classification head
class ClassificationHead(nn.Module):

    def __init__(self, numClasses, dropout=0.3, point_dimension=3):
        super(ClassificationHead, self).__init__()
        self.basePointNet = BasePointNet(returnLocalFeatures=False, pointDimension=point_dimension)

        self.Linear = nn.Sequential(
            

          nn.Linear(1024, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),

          nn.Linear(512, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(),

          nn.Dropout(dropout),
          
          nn.Linear(256, numClasses)


        )

    def forward(self, x):
        x, featureTransform = self.basePointNet(x)

        for index,layer in enumerate(self.Linear):
          x = layer(x)

        return F.log_softmax(x, dim=1), featureTransform


#segmentation head
class SegmentationHead(nn.Module):

    def __init__(self, numClasses, pointDimension=3):
        super(SegmentationHead, self).__init__()
        self.base_pointnet = BasePointNet(returnLocalFeatures=True, pointDimension=pointDimension)

        self.Convolutions = nn.Sequential(
          nn.Conv1d(1088, 512, 1),
          nn.BatchNorm1d(512),
          nn.ReLU(),

          nn.Conv1d(512, 256, 1),
          nn.BatchNorm1d(512),
          nn.ReLU(),

          nn.Conv1d(256, 128, 1),
          nn.BatchNorm1d(128),
          nn.ReLU(),

          nn.Conv1d(128, numClasses, 1)
        )


    def forward(self, x):
        x, featureTransform = self.BasePointNet(x)

        x = x.transpose(2, 1)
        for index,layer in enumerate(self.Convolutions):
          x = layer(x)


        x = x.transpose(2, 1)

        return F.log_softmax(x, dim=-1), featureTransform

