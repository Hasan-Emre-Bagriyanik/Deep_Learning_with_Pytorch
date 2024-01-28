# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:50:15 2023

@author: Hasan Emre
"""

#%% import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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
n_iters =  5000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets 
train = TensorDataset(featuresTrain, targetsTrain)
test = TensorDataset(featuresTest, targetsTest) 

# data loader 
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)


#%% Create RNN Model

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity = "relu")
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    

        
# Create RNN
input_dim = 28
hidden_dim = 100
layer_dim = 2
output_dim = 10

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)

#%% RNN Model Training 

seq_dim = 28
loss_list = []
iteration_list = []
accuracy_list = []
count = 0

for epochs in range(num_epochs):
    for i ,(images, labels) in enumerate(train_loader):
        
        train = Variable(images.view(-1, seq_dim, input_dim))
        labels = Variable(labels)
        
        # clear Gradients
        optimizer.zero_grad()
        
        # forward propagation
        outputs = model(train)
        
        # calculate softmax and cross entropy loss
        loss = error(outputs, labels)
        
        # calculating gradients
        loss.backward()
        
        # udpate parameters
        optimizer.step()
        
        count += 1
        
        
        if count % 250 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            
            #iterate through test dataset
            for images, labels in test_loader:
                
                images = Variable(images.view(-1, seq_dim, input_dim))
                
                # forward propagation
                outputs = model(images)
                
                # get Predictionsfrom the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # total number of labels
                total += labels.size(0)
                
                correct += (predicted == labels).sum()
                
            accuracy =100* correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            
            if count % 500 == 0:
                print("Iteration: {}  Loss: {}  Accuracy: {} %".format(count, loss.item(), accuracy))
                

#%% Visualization

# Visualization Loss
plt.plot(iteration_list, loss_list, color = "red")
plt.xlabel("Number Of Iteration")
plt.ylabel("Loss")
plt.title("RNN: Loss VS Number Of Iteration")
plt.show()

plt.figure()

# Visualization Accuracy
plt.plot(iteration_list, accuracy_list, color = "blue")
plt.xlabel("Number Of Iteration")
plt.ylabel("Accuracy")
plt.title("RNN: Accuracy VS Number Of Iteration")
plt.show()

