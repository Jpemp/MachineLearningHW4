from ucimlrepo import fetch_ucirepo 
import pandas as pd
import math

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

def standard_dev(X, X_m):
    total = len(X)
    var = (1/(total-1)) * ((X-X_m) ** 2)
    s_var = var.sum()
    sd = (s_var ** 0.5)
    return sd

X_m = mean(X)

X_sd = standard_dev(X, X_m)

X_z = (X-X_m)/X_sd

#Display X and Y after Z-score normalization
print(X_z)
print(Y)

#Problem 2

def softmax():
    print(1)
    return 0

def kfold_split():
    print(2)
    return 0

def accuracy():
    print(3)
    return 0

def confusion_matrix():
    print(4)
    return 0
