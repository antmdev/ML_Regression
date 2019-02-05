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
#Building the optimal model using Backward Elimination
"""
import statsmodels.formula.api as sm    #import statsmodellibrary
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) #add matrix of ones to X

#create the OPTIMAL matrix team that only contains high impact independnt variables
X_opt = X[:, [0, 1, 2, 3, 4, 5]] 
#grab the OLS (least squares) method
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#use preiction to understand P value >SL or not.
regressor_OLS.summary()
"""
#X2 had the highest Pvalue - 
so remove this and repeat the proces of fitting the curve
X2 dummy variable for state (location)
"""
X_opt = X[:, [0, 1, 3, 4, 5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
"""
#X1 Remove the X1 value (Other dummy State remaining variable)
"""
X_opt = X[:, [0, 3, 4, 5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
"""
#X1 Remove the X2 value (Which correlates to Administration)
"""
#This was 60% value essentially saying it hs no impact on the profit
X_opt = X[:, [0, 3, 5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
"""
#X1 Remove the X2 value (Which correlates to Marketing SPend)
"""
#This was 60% value essentially saying it hs no impact on the profit
X_opt = X[:, [0, 3,]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Backward Elimination has indicated that  
Marketing spend is the most significant independent variable
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Re-run prediction

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #Create object of linear regressoin class
regressor.fit(X_train, y_train) #apply the fit method of teh Multiple linear Regression Algorythm to the training set
"""
#Predicting the Test Set Results
"""
y_pred = regressor.predict(X_test)


    

    





















