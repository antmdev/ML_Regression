# Simple Linear Regression
"""
# Data Preprocessing Template
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
#Importing the dataset
"""
dataset = pd.read_csv('pwd.csv')
X = dataset.iloc[:, :-1].values#setting independent variable 
y = dataset.iloc[:, 1].values  #settign dependent variable 

"""
#Splitting the dataset into the Training set and Test set
"""
from sklearn.model_selection import train_test_split #library to split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

"""
#Fitting the Simple Linear Regression to the Training Set
"""
from sklearn.linear_model import LinearRegression #import LR class
regressor = LinearRegression() #Create and Object of the linear regression class called regressor
regressor.fit(X_train, y_train)    #use the fit method to fit the training set

"""
#Predicting the Test set Results
"""
y_pred = regressor.predict(X_test)

"""
#Visualising the Training set Results! Woop!
"""
plt.scatter(X_train, y_train, color = 'red') #real year vs salary
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #plotting the prediction line of the ML model
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

"""
#Visualising the Test set Results! Woop!
"""
plt.scatter(X_test, y_test, color = 'red') #real year vs salary
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #plotting the prediction line of the ML model
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
