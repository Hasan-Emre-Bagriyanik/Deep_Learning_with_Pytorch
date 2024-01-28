# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 14:22:30 2023

@author: Hasan Emre
"""

#%% import library

import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#%% Prepare Dataset

# load data
train = pd.read_csv("mnist_train.csv", dtype = np.float32)
print(train.head())

# split data into features(pixels) and labels (numbers from 0 to 9)
targets_numpy = train.label.values
features_numpy = train.iloc[:,train.columns != "label"].values/255  # Normalization

# train test split. Size of train data is %80 and size of size of test is %20
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy, targets_numpy, test_size=0.2, random_state=42)


# create features and targets tensor for train set. As you remember we need variable to accumulate gradients.  
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)  # data type is long

# create features and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)  # data type is long

# batch_size, epochs and iteration
batch_size = 100
n_iters =  10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets 
train = TensorDataset(featuresTrain, targetsTrain)
test = TensorDataset(featuresTest, targetsTest) 

# data loader 
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

# visualize one of the images in dataset
plt.imshow(features_numpy[10].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[10]))
plt.savefig("graph.png")
plt.show()

#%% Create Logistic Regression Model

# create class
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super function. It inherits from nn.Module adn we can access everythink in nn.Module
        super(LogisticRegressionModel, self).__init__()
        # Linear part
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)
    

# Instantiate Model Class
input_dim = 28*28 # size of image px*px
output_dim = 10  # labels

# create logistic regression model 
model = LogisticRegressionModel(input_dim, output_dim) 

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer 
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


#%% Training the Model

count = 0
loss_list = []
iteration_list = []

for epochs in range(num_epochs):
    for i ,(images, labels) in enumerate(train_loader):
        
        # Define variables
        train = Variable(images.view(-1,28*28))
        labels = Variable(labels)
        
        # clear gradients 
        optimizer.zero_grad()
        
        # forward to get output
        outputs = model(train)
        
        # calculate softmax and cross entropy loss
        loss = error(outputs, labels)
        
        # Calculate gradients 
        loss.backward()
        
        # updating parameters
        optimizer.step()
        
        count += 1
        
        # Prediciton
        if count % 50 == 0:
            # calculate Accuracy 
            correct  = 0
            total = 0
            # Predict test dataset 
            for images, labels in test_loader:
                test = Variable(images.view(-1, 28*28))
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value 
                predicted = torch.max(outputs.data, 1)[1]
                
                # total number of labels
                total += len(labels)
                
                correct += (predicted == labels).sum()
                
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration 
            loss_list.append(loss.data)
            iteration_list.append(count)
        
        if count % 500 == 0:
            # print Loss
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
            
            
#%% Visualization
plt.plot(iteration_list, loss_list)
plt.xlabel("Number of Iteration")
plt.ylabel("Loss")
plt.title("Logistic Regression: Loss VS Number of Iteration")
plt.show()
            
            
