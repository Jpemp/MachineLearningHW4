from ucimlrepo import fetch_ucirepo 
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

#Problem 1
#Dataset fetching
wine = fetch_ucirepo(id=109)

#Loaded as pandas dataframes
X = wine.data.features
Y = wine.data.targets

#Limiting data to the first 40 samples
X = X.head(40)
Y = Y.head(40)

#Display X and Y
print(X)
print(Y)

def mean(X):
    total = len(X)
    mean = (1/total) * X.sum()
    return mean

def standard_dev(X,X_m):
    total = len(X)
    var = (1/(total-1)) * ((X-X_m) ** 2)
    s_var = var.sum()
    sd = (s_var ** 0.5)
    return sd

#Z-score calculation
X_m = mean(X)
X_sd = standard_dev(X, X_m)
X_z = (X-X_m)/X_sd

#Display X after Z-score normalization, along with Y
print(X_z)
print(Y)

#Problem 2

def softmax(X):
    e = math.exp(X)
    soft = e/e.sum()
    return soft

def kfold_split(X, k):
    X_5_split = np.array_split(X,k)
    return X_5_split

def accuracy(Y_true, Y_pred):
    accurate_pred = 0
    total = len(Y_true)

    for i in range(total):
        if Y_true[i] == Y_pred[i]:
            accurate_pred = accurate_pred + 1
    acc = accurate_pred/total
    return acc

def confusion_matrix():
    

    return c_m

ks = kfold_split(X,5) 

#Problem 3

#Problem 4
