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
    split_size = len(X)//k
    X_5_split = []
    section = []
    for i in range(k):
        start = i * split_size
        end = split_size + (i * split_size)
        section = X[start:end]
        X_5_split.append(section)
    return X_5_split

def accuracy(Y_true, Y_pred):
    accurate_pred = 0
    total = len(Y_true)

    for i in range(total):
        if Y_true[i] == Y_pred[i]:
            accurate_pred = accurate_pred + 1
    acc = accurate_pred/total
    return acc

def confusion_matrix(Y, Y_pred):
    #for i


    return c_m 

ks = kfold_split(X_z, 5)
print(ks)

#Problem 3

#To split train and test data
def batch(ks, number):
    train = []
    test = []
    for i in range(len(ks)):
        if i == number:
            test = ks[i]
        else:
            train.append(ks[i])
    train = pd.concat(train)
    return train, test

k1_train, k1_test = batch(ks, 0)
k2_train, k2_test = batch(ks, 1)
k3_train, k3_test = batch(ks, 2)
k4_train, k4_test = batch(ks, 3)
k5_train, k5_test = batch ks, 4)


#Problem 4
