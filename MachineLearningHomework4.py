from ucimlrepo import fetch_ucirepo 
import pandas as pd

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

X = (X-X.mean())/X.std()

#Display X and Y after Z-score normalization
print(X)

#Problem 2
