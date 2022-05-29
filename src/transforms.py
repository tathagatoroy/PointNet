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

# This transform is to sample points from the faces to generate a uniformly distributed pointcloud 
class PointSampler(object):

  # the input samplen is object with 3 keys : 'pos' ,'face' and 'y'
  # all the objects are precomputed to tensor
  # for some stupid reason faces have shape 3,l instead of l,3

  #initialise with numberof points you want to sample
  def __init__(self,numPoints):
    #check if this is an instance of int
    assert isinstance(numPoints,int)
    self.numPoints = numPoints


  #find the area of a face
  def computeFaceArea(self,face,vertices):

    #print(vertices.shape)
    #print(face[0])

    pt1 = vertices[face[0]]
    pt2 = vertices[face[1]]
    pt3 = vertices[face[2]]
    #compute area using herons formula = root(s(s-a)(s-b)(s-c))
    a = np.linalg.norm(pt1-pt2)
    b = np.linalg.norm(pt2-pt3)
    c = np.linalg.norm(pt3-pt1)
    s = (a + b + c) / 2
    
    area = s*(s - a)*(s - b)*(s - c)
    if area < 0:
      area = 0
    return area ** 0.5

  #compute probabilities of sampling each faces proportional with area
  # returns a list not an numpy array
  def computeFaceProbabilities(self,faces,vertices):
    face_probabilities = []
    l,b = faces.shape # l = number of faces and b is 3
    normalizing_factor = 0 # sum of all probabilities that can be used to scale probabilities between 0 and 1
    for i in range(l):
      face = faces[i]
      face_probability = self.computeFaceArea(face,vertices)
      normalizing_factor += face_probability
      face_probabilities.append(face_probability)
    for i in range(l):
      face_probabilities[i] = face_probabilities[i] / normalizing_factor
    return face_probabilities


  ''' sample point from a given face '''
  def sampleFacePoint(self,face,vertices):

    #return

    #get the 3 coordinates 
    pt1 = vertices[face[0]]
    pt2 = vertices[face[1]]
    pt3 = vertices[face[2]]

    # there is something called baryocentric coordinates 
    # given a triangular face with vertices A,B,C
    # any point on the face can be represented by p = aA + bB + cC such that a + b + c = 1
    random.seed(time.time())

    a,b,c = random.random() , random.random() , random.random()
    sum = a + b + c
    a = a / sum
    b = b / sum
    c = c / sum
    new_pt = []
    for i in range(3):
      new_pt.append(a * pt1[i] + b * pt2[i] + c * pt3[i])

    return new_pt


  #output 3000 sampled point from the mesh given by the object file
  def __call__(self,mesh):
    #compute the probabilities of sampling each face
    faces, vertices = mesh['face'],mesh['pos'].T
    outputClass = mesh['y']
    #convert to numpy for safety
    # this is actually a bad policy in general as it is better to work with tensors.
    # but I wrote the code already assuming this would be numpy
    # I am unwilling to change the code now.Convert back to tensor in the form (numPoints, 3) before returning 
    faces = faces.numpy()
    vertices = vertices.numpy()
    #print(faces.shape)
    #print(vertices.shape)
    #for some reason the np.array are in transposed.Need to undo that
    faces = faces.T
    vertices = vertices.T
    faceProbabilities = self.computeFaceProbabilities(faces,vertices)

    #choose the face with replacement with proportionality of their weights
    sampledFaces = random.choices(faces,weights = faceProbabilities, cum_weights = None , k = self.numPoints )
    sampledPoints = []
    for i in range(self.numPoints):
      # get a point in the face
      pt = self.sampleFacePoint(sampledFaces[i],vertices)
      sampledPoints.append(pt)
    return (torch.from_numpy(np.array(sampledPoints)),outputClass) # output is a N * 3 pointcloud tensor

  #new_vertices = pointCloudSampler(3000,vertices,faces)
  #print(new_vertices.shape)
  #new vertices is 3000 * 3 pointcloud
  #visualizePointClouds(new_vertices)




#transform class to normalize the dataset that is zero centering and scaling it to a unit sphere 
class Normalize(object):
  # the class normalized would be applied after sampling points.
  #So the input is a tensor 
  def __call__(self,sample):
    # normalized the point cloud by subtracting its mean and scaling it to a unit sphere
    #assume the input is N * 3 point cloud tensor
    #get the mean
    #convert to numpy 
    sample,outputClass = sample
    sample = sample.numpy()
    #print(sample.shape)

    mean = np.mean(sample,axis = 0)
    #subtract the mean
    sample = sample - mean
    #find the largest magnitude distance among the points
    farthest_distance = np.max(np.linalg.norm(sample,axis = 1))
    #scale using the distance to project the cloud inside a sphere of radius 1
    sample = sample / farthest_distance
    return (torch.from_numpy(sample),outputClass)




# class to perform to random rotation about Z 
class RandomRotateZ(object):
  #performs the rotation with a random angle theta
  # this happens after normalize hence input is N * 3
  def __call__(self,sample):
    theta = random.random() * 2. * math.pi
    sample,outputClass = sample
    tmp = sample.T
    rot = np.array([
                [np.cos(theta),-np.sin(theta),0],
                [np.sin(theta),np.cos(theta),0],
                [0,0,1]

            
            ])
    #print(torch.is_tensor(sample))

    newPointcloud = rot.dot(tmp).T
    return (torch.from_numpy(newPointcloud),outputClass)


