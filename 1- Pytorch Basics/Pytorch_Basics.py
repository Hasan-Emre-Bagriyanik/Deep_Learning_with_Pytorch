# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 11:38:33 2023

@author: Hasan Emre
"""

#%%  Import Library

import torch
import numpy as np
from torch.autograd import Variable

#%%  Matrix 

# Pytorch array
array = [[1,2,3],[4,5,6]]
tensor = torch.Tensor(array)

print("Array Type: ", (tensor.type))  # type
print("Array Shape: {}".format(tensor.shape))  # shape
print(tensor)

#%% Pytorch ones  (Birlerden olusan bir matrix)

print(torch.ones((2,3)))

#%% Pytorch random  (Rastgele sayilardan matrix olusturur (0'la 1 araliginda))
print(torch.rand(2,3))

#%% (Pytorch ve numpy donusumleri)

# random numpy array  
array = np.random.rand(2,2)
print("{} {}\n".format(type(array), array))

# from numpy to tensor
from_numpy_to_tensor = torch.from_numpy(array)
print("{}\n".format(from_numpy_to_tensor))

# from tensor to numpy
tensor = from_numpy_to_tensor
from_tensor_to_numpy = tensor.numpy()
print("{} {}\n".format(type(from_tensor_to_numpy), from_tensor_to_numpy))



#%%  Basic Math with Pytorch

# create tensor
tensor = torch.ones((3,3))
print("\n{}\n".format(tensor))

# Resize
print("{}\n{}\n".format(tensor.view(9).shape, tensor.view(9)))

# Addition (Ekleme islemi)
print("Addition: {}\n".format(torch.add(tensor,tensor)))

# Subtraction  (Cikarma islemi)
print("Subtracttion: {}\n".format(tensor.sub(tensor)))

# Element wise multiplication  (iki matrixi carpimi)
print("Element wise multiplication: {}\n".format(torch.mul(tensor, tensor)))

# Element wise division  (iki matrixin bolumu)
print("Element wise division: {}\n".format(torch.div(tensor,tensor)))

# Mean  ( Ortalama hesapliyor)
print("Mean: {}\n".format(tensor.mean()))

# Stanadart deviation (std) ( standart sapma hesapliyor)
print("std: {}\n".format(tensor.std()))



#%% Variables (Degiskenler)

# define variable
var = Variable(torch.ones(3), requires_grad = True)
print(var, "\n")

# Lets make basic backward propagation
# We have an equation that is y = x^2
array = [2,4]
tensor = torch.Tensor(array)
x = Variable(tensor, requires_grad = True)
y = x**2
print("y = ", y)

# recap o equation o = 1/2*sum(y)
o = (1/2)*sum(y)
print("o = ", o)

# backward
o.backward()  # claculates gradients 
print("gradients: ", x.grad)


