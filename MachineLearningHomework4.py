from ucimlrepo import fetch_ucirepo 
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import time

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
def GDS(X_train, Y_train, learning_rate, epochs, batch_size):
    weights = np.zeros((X_train.shape[1], 1))
 
    #Data shuffle
    for i in range(epochs):
        #mini batch creation
        mini_batches = []
        train_data  = np.hstack((X_train, Y_train))
        shuffle = np.random.permutation(train_data)
        n_batch = shuffle.shape[0] // batch_size
        j=0
        for j in range(n_batch + 1):
            batch = train_data[j*batch_size:(j+1)*batch_size, :]
            X = batch[:,:-1]
            Y = batch[:,-1].reshape((-1,1))
            mini_batches.append((X, Y))
        if train_data.shape[0] % batch_size != 0:
            batch = train_data[j*batch_size:train_data.shape[0]]
            X = batch[:,:-1]
            Y = batch[:,-1].reshape((-1,1))
            mini_batches.append((X, Y))

        for k in mini_batches:
            X_b,Y_b = k
            #Prediction computation
            y_pred = np.dot(X_b, weights)


            #Gradient computation
            gradient = np.dot(X_b.T, (y_pred-Y_b))

            #Weight updates
            weights -= learning_rate*gradient

        print (time.time())
    return weights

#10 K folds
X_k = kfold_split(X_z, 10) #This produces a list of 10 K folds. Each K fold is a tuple, the first item is the train data and the second item is the test data. 
Y_k = kfold_split(Y, 10)   #To access the test data of the first k fold, for example, use ks[0][1]. This looks at the second item of the first list. First index goes through the k-folds, second index is the tuple.
print(Y_k)
#Get weights for each K fold
y_pred = []
for i in range(len(X_k)):
    W = GDS(X_k[i][0], Y_k[i][0], 0.1, 500, 36)

    y_pred = np.dot(X_k[i][1], W)

#Problem 4
