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
    X_split = []
    train = []
    test = []
    k_folds = []
    for i in range(k):
        start = i * split_size
        end = split_size + (i * split_size)
        X_split.append(X[start:end])
    print(X_split)

    for i in range(k):
        test = X_split[i]
        for j in range(k):
            if j != i:
                train.append(X_split[j])
        
        train_pd = pd.concat(train)
        k_folds.append([train_pd,test])
        train_pd = train_pd.iloc[0:0]
        train.clear()
    return k_folds

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



#Problem 3

#10 K folds
ks = kfold_split(X_z, 10) #This produces a list of 10 K folds. Each K fold is a tuple, the first item is the train data and the second item is the test data. 
                          #To access the test data of the first k fold, for example, use ks[0][1]. This looks at the second item of the first list. First index goes through the k-folds, second index is the tuple.

def GDS():
    return 0

SGD_Step_Size = 0.1
Epochs = 500



#Problem 4
