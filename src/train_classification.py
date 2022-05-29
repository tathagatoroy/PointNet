#importing the neccesary libraries


import numpy as np
#import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms 
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
import argparse
import wandb
import logging
import time
from  torchsummary import summary
#initialise wandb project
wandb.init(project = "PointNetClassification", entity = "roy3")


#importing transforms and models
from transforms import PointSampler , RandomRotateZ , Normalize
from model import ClassificationHead

#accepting command line parameters using argparse
parser = argparse.ArgumentParser()

#specify the parameters allowed
parser.add_argument('--batchsize', type = int , default = 32 , help = 'provide the batchsize(dedault is 32 ')
parser.add_argument('--dataset', type = int , default = 10 , help = 'provide the dataset you want to train, 10 for modelnet10 and 40 for modelnet 40')
parser.add_argument('--learningRate', type = float , default = 0.001 , help = ' provide the learning rate default is 0.001')

#parser.add_argument('--dropout ', type = float , default = 0.3 , help = ' provide the dropout value between 0 and 1, default is 0.3')
parser.add_argument('--epochs', type = int , default = 15, help = ' number of epochs you want to run')
parser.add_argument('--model', type = str , help = 'provide model path if you want to train a existing model')
parser.add_argument('--log',type = str , default = './logs/log.txt', help = ' filepath where you want the logs to be ')
parser.add_argument('--numPoints', type = int , default = 1024 , help = "number of sampled points in the point cloud ")

opt = parser.parse_args()
#generate the wandb config parameters
wandb.config.learningRate = opt.learningRate 
#print(opt)

#print(opt.dropout)
#wandb.config.dropout = opt.dropout 

wandb.config.epochs = opt.epochs
wandb.config.loss = "Cross Entropy"
wandb.config.optimizer = "Adam"
wandb.config.numPoints = opt.numPoints

#setting up numpoints 
numPoints = opt.numPoints
#setting up logging and model settings
isLog = False
isModel = False
modelPath = None
logPath = None

#set the Model variables
if opt.model is not None :
        isLog = True
        isModel = opt.model
#set the log paramters 
if opt.log is not None:
        isLog = True
        logPath = opt.log 

logging.basicConfig(level = logging.INFO, filename = logPath , filemode = 'w')
#function which does logs, I dont wanna repeatedly check the isLog variable 
def log(output,isLog):
        if isLog:
                logging.info(output)
        print(output)


#set the batch size , optimizer , criterion , Dataset name , dropout , learning rate, num_epochs 
#set batchsize
BATCH_SIZE = opt.batchsize

#set dataset type and name 
# curently only ModelNet10 or ModelNet40 , might add Shapenet later
DATASET_NAME = "ModelNet10"
datasetParameter = "10" # to pass it to the pytorch dataloader
if opt.dataset == 40:
        DATASET_NAME = "ModelNet40" 
        datasetParameter = "40"


#set dropout , num of epochs and Learning rate
#DROPOUT = opt.dropout
LEARNING_RATE = opt.learningRate
NUM_EPOCHS = opt.epochs






Transforms = torchvision.transforms.Compose([PointSampler(numPoints),
                    Normalize(),
                    RandomRotateZ()
                     ])

#Load the dataset
log("The Train and Test Dataset is being loaded",isLog)
tt = time.time()
trainDataset = ModelNet(root = "./../data/ModelNetTrain", name = datasetParameter ,train = True , transform = Transforms)
log("training and validation set loaded",isLog)
log("The size of training + validation set is " + str(len(trainDataset)),isLog)
tt1 = time.time()
log("Time taken to load the train set : " + str(tt1 - tt),isLog)
testDataset = ModelNet(root = "./../data/ModelNetTest",name = datasetParameter, train = False, transform = Transforms)
log("test set is being loaded ",isLog)
log("The size of test set is " + str(len(testDataset)),isLog)
tt2 = time.time()
log("Time taken to load the test set : " + str(tt2 - tt1),isLog)



#print(len(trainDataset))
#print(len(testDataset))



#perform dataloading in batches
trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)



log("Data Loading is finished",isLog)

#place where we checkpoint the model
saveModelPath = "./../models/classificationModel_" + DATASET_NAME + ".pt"

#divide train into train and val
NUM_BATCHES = len(trainDataLoader)
TRAIN_BATCHES = int (0.8 * NUM_BATCHES)
VAL_BATCHES = NUM_BATCHES - TRAIN_BATCHES


#load the classification model

#initialize model,optimizer and  criterion
model = None
if datasetParameter == "10":
    model = ClassificationHead(10)
else:
    model = ClassificationHead(40)
if isModel :
        model.load_State_dict(torch.load(modelPath))



        


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

#setting up the gpu 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#check whether gpu is found or not
if device == "cpu":
        log("No GPU",isLog)
else:
        log("GPU found ", isLog)
    
#storing all loss functions and accuracy along for train , test and val
#all of this will be logged using weights and biases 

error = { "train" : [] , "test" : [] , "val" : [] }
accuracy = { "train" : [] , "test" : [] , "val" : [] }

# load the model to device and set up train mode
model.train()
model = model.float()
model = model.to(device)
#log the model summary
#log(summary(model,(3,numPoints)),isLog)




#start training 

#print Model Summary using pytorch

#pointClouds, labels = next(iter(trainDataLoader))
#pointClouds = pointClouds.transpose(2,1)
#print(pointClouds.shape)
log("The device is " + str(torch.cuda.current_device()) + " and the device name is " + str(torch.cuda.get_device_name(0)),isLog)
for epoch in range(NUM_EPOCHS):
        log("Starting epoch : " + str(epoch + 1) , isLog)
        trainSamples = 0.0
        valSamples = 0.0
        trainLoss = 0.0 
        valLoss = 0.0
        valAccuracy = 0.0
        trainAccuracy = 0.0
        t1 = time.time()
        for i,data in enumerate(trainDataLoader): # data has the format (input,labels
                #check if its training or val set 
                if i < TRAIN_BATCHES:
                        #trainSet 
                        # returns data = [BATCHSIZE, NUMPOINTS, 3], [BATCHSIZE,1]
                        (pointClouds,labels) = data 
                        #pytorch needs B,C,N not B,N,C

                        #add the data to GPU
                        pointClouds = pointClouds.transpose(2,1)
                        #pointClouds = pointClouds.to(torch.float64)
                        
                        pointClouds = pointClouds.to(device)
                        # print(pointClouds.dtype)
                        pointClouds = pointClouds.type(torch.float32)
                        labels = labels.to(device)
                        
                    
                        #print(labels)
                        #forward pass 
                        logits,features = model(pointClouds)
                        #print(logits.shape)
                        #compute the loss 
                        #print(logits.shape)
                        labels = labels.reshape(labels.shape[0])
                        #print(labels.shape)
                        loss = criterion(logits,labels)

                        #set the gradients to zero
                        optimizer.zero_grad()
                        #back prop 
                        loss.backward()

                        #update parameters
                        optimizer.step()
                        #update loss 
                        trainLoss += loss.item()

                        #compute the predictions
                        predictions = logits.argmax(dim = 1)

                        #update total correct 
                        trainAccuracy += (predictions == labels).sum()

                        
                        #update count for number of samples 
                        trainSamples += logits.shape[0]
                        print("Progress : {0} ".format(trainSamples / (TRAIN_BATCHES * BATCH_SIZE)))
                else:
                        #log("Validation starting ",isLog)
                        #model.evals()
                        #val set
                        # returns data = [BATCHSIZE, NUMPOINTS, 3], [BATCHSIZE,1]
                        (pointClouds,labels) = data
                        #pytorch needs B,C,N not B,N,C
                        #add the data to GP
                        pointClouds = pointClouds.transpose(2,1)
                        pointClouds = pointClouds.to(device)
                        pointClouds = pointClouds.type(torch.float32)
                        labels = labels.to(device)

                        #forward pass
                        logits,features = model(pointClouds)
                        #compute loss
                        labels = labels.reshape(labels.shape[0])
                        loss = criterion(logits,labels)

                        #compute val loss 
                        valLoss += loss.item()

                        #compute predictions 
                        predictions = logits.argmax(dim = 1)

                        #update total correct 
                        valAccuracy += (predictions == labels).sum()

                        #update the size of the valSamples
                        valSamples += logits.shape[0]
                        print("Progress : {0} ".format(valSamples / (len(trainDataset) - TRAIN_BATCHES * BATCH_SIZE)))

                        

        #epoch done need to scale the values 
        trainLoss = torch.div(trainLoss , trainSamples)
        valLoss = torch.div(valLoss ,valSamples)
        trainAccuracy = torch.div(trainAccuracy  ,trainSamples)
        valAccuracy = torch.div(valAccuracy ,valSamples)
        
        #log the results 
        log(" Epoch : " + str(epoch + 1) + " : train Loss : " + str(trainLoss) + " train Accuracy " + str(trainAccuracy) + "val loss : " + str(valLoss) + " val Accuracy : " + str(valAccuracy) , isLog)

        #add the error to history 
        error['train'].append(trainLoss)
        error['val'].append(valLoss)

        #add the accuracy to history 
        accuracy['train'].append(trainAccuracy)
        accuracy['val'].append(valAccuracy)



        #log into wandb , hope this works
        wandb.log({'train accuracy' : trainAccuracy , 'validation accuracy ' : valAccuracy})
        wandb.log({'train loss' : trainLoss , 'validation loss ' : valLoss})
        
        #log the time take for training per epoch
        t2 = time.time()

        log("Time taken for epoch " + str(epoch + 1) +  " is : " + str(t2 - t1) , isLog)

        #if epoch is divisible by 5, save the model
        if (epoch + 1) % 5 == 0 :
                log("saving the model .... ",isLog)
                torch.save(model.state_dict(), saveModelPath)
                


#save the model one last time 
log("saving the model .....",isLog)
torch.save(model.state_dict(),saveModelPath)
log("Training Done " , isLog)

#Testing 
model = model.to(device)
model.eval()

testLoss = 0.0
testAccuracy = 0.0
testSamples = 0.0
log("Testing now ",isLog)
for i,data in enumerate(testDataLoader):
        #log("Testing begins ",isLog)
        (pointClouds,labels) = data 
        pointClouds = pointClouds.transpose(2,1)
        pointClouds = pointClouds.to(device)
        pointClouds = pointClouds.type(torch.float32)
        labels = labels.to(device)

        #output 
        logits,features = model(pointClouds)
        predictions = logits.argmnax(dim = 1)
        
        #compute accuracy 
        testAccuracy += (predictions == labels).sum()

        #compute loss
        loss = criterion(logits,labels)

        testLoss += loss.item()
        testSamples += labels.shape[0]
        print("Progress : {0} ".format(testSamples / len(testDataset)))

#normalize the loss
testLoss = torch.div(testLoss , testSamples)
trainLoss = torch.div(trainLoss, testSamples)

#log the output 
log("The test loss : " + str(testLoss) + " and the test Accuracy " + str(testAccuracy),isLog)
wandb.log({"test loss " : testLoss , "test accuracy " : testAccuracy})


