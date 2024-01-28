# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:28:37 2023

@author: Hasan Emre
"""

#%% import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from torch.autograd import Variable

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


#%% Create ANN Model

class ANNModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        
        # lienar function 1: 784 --> 150
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-Linearity 1 
        self.relu1 = nn.ReLU()
        
        # lienar function 2: 150 --> 150
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-Linearity 2 
        self.tanh2  =nn.Tanh()

        # lienar function 3: 150 --> 150
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-Linearity 3
        self.relu3 = nn.ReLU()
            
        # lienar function 4 (readout): 150 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)
    
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.tanh2(out)
        
        # Linear function 3
        out = self.fc3(out)
        # Non-linearity 3
        out = self.relu3(out)
        
        # Linear function 4
        out = self.fc4(out)
        
        return out
    
# instantiate ANN 

input_dim = 28*28
hidden_dim = 150 # Hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason
output_dim = 10

# Create ANN
model = ANNModel(input_dim, hidden_dim, output_dim)

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr= 0.02)


#%% ANN Model Training

count = 0
loss_list = []
iteration_list = []
accuracy_list = []

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
            
            # store loss, accuracy and iteration 
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if count % 500 == 0:
            # print Loss
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
                
#%% Visualization

# Visualization Loss
plt.plot(iteration_list, loss_list, color = "red")
plt.xlabel("Number Of Iteration")
plt.ylabel("Loss")
plt.title("ANN: Loss VS Number Of Iteration")
plt.show()

plt.figure()

# Visualization Accuracy
plt.plot(iteration_list, accuracy_list, color = "blue")
plt.xlabel("Number Of Iteration")
plt.ylabel("Accuracy")
plt.title("ANN: Accuracy VS Number Of Iteration")
plt.show()


