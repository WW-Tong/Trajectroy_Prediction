import pickle
import os
import torch
import numpy

path="D:\GitHub_items\Trajectron\sgan-dataset\data\eth_train.pkl"
path="D:\Project\Cifar10_100\\net_params.pkl"
with open(path, 'rb') as handle:
    new_fet = pickle.load(handle, encoding='bytes')
t = torch.from_numpy(new_fet)
print(t.shape)