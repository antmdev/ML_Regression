# Multiple Linear Regression
"""
# Importing the libraries
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
# Importing the dataset
"""
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

"""
# Encoding categorical data
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

"""
# Avoiding the Dummy Variable Trap
"""
X = X[:, 1:] 
#Get rid of just the first column which is california
#but the ML algortihm actually already does this in MLRregression

"""
# Splitting the dataset into the Training set and Test set
"""
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#had to use a different version of SKLEARN hear? not compatible for soem reason
#from sklearn.cross_validation - cannot be importred???
"""
#Fitting Multiple Linear Regression to teh Trainig Set
"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #Create object of linear regressoin class
regressor.fit(X_train, y_train) #apply the fit method of teh Multiple linear Regression Algorythm to the training set
"""
#Predicting the Test Set Results
"""
y_pred = regressor.predict(X_test)

"""
#Calculate the Average error margin for predicted results (ANT CODE)
"""

#error_margin = ((y_test[6] - y_pred[6]) / y_pred[6]) * 100

for y_pred in y_pred:
    print(y_pred)
    

    

    





















