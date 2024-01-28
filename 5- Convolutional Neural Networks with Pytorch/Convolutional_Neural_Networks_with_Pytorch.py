# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:32:43 2023

@author: Hasan Emre
"""

#%% import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn  as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader ,TensorDataset
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


#%% Create CNN Model

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        #è Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully Connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10)
        
    def forward(self, x):
        # Convolution 1 
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
 
        # Max pool 2
        out = self.maxpool2(out)
        
        # Flatten the tensor before passing through the fully connected layer
        out = out.view(out.size(0), -1)
        
        # linear function (readout)
        out = self.fc1(out)
        
        return out
    
# Create model
model = CNNModel()

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)


#%%  CNN Model Training

count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epochs in range(num_epochs):
    for i ,(images, labels) in enumerate(train_loader):
        
        # Define variables
        train = Variable(images.view(100,1,28,28))
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
                test = Variable(images.view(100,1,28,28))
                
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
plt.title("CNN: Loss VS Number Of Iteration")
plt.show()

plt.figure()

# Visualization Accuracy
plt.plot(iteration_list, accuracy_list, color = "blue")
plt.xlabel("Number Of Iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy VS Number Of Iteration")
plt.show()


