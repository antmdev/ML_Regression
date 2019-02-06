# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
# Importing the dataset
"""
dataset = pd.read_csv('Position_Salaries.csv')
#using 1:2 means that X is now considered as a matrix rather than a vector
#I>E MAtrix (10,0) and Y is a vector (10) just 10 rows.
X = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values

"""
# Splitting the dataset into the Training set and Test set
"""
#No need to split into trainng and test ast there's only 10 values so theres not enough
#data to properly train and test.
#Also we need to make an extremely accurate prediction because we're tying to 
#Predict where the employees salary actually sits... so we want to use all of the data.
"""
# Feature Scaling
"""
#No need for Feature scaling as we're using the same linear regressoin library that
#alredy includes feature scaling
"""
# Build the Linear Regression Model #so we can compare to PLR
"""
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
"""
# Build the Polynomial Linear Regression Model #so we can compare to PLR
"""
#create the X_poly matrix first
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
#Transform X matrix of features containig only the position levels in to the X_poly  
X_poly = poly_reg.fit_transform(X) 
#Then create a second regression and fit to the new X_poly Set
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

"""
# Visualise the Linear Regression result
"""
plt.scatter(X, y, color = 'red') #Base results Level Vs Salary from Company
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""
# Visualise the Polynomial Linear Regression result
"""
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red') 
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""
#Predict a new result with linear regression
"""
lin_reg.predict(np.array(6.5).reshape(1, -1))

"""
#Predict a new result with Polynomial linear regression
"""
lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape(1, -1)))









