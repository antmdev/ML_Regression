# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
# Importing the dataset
"""
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values

"""
# Splitting the dataset into the Training set and Test set
"""
#from sklearn.model_selection import train_test_split #library to split train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


"""
# Feature Scaling
"""
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler() 
# X_train = sc_X.fit_transform(X_train) 
# X_test = sc_X.transform(X_test) 

"""
# Fitting the Regression Model to the Dataset
"""


"""
#Predict a new result with Polynomial linear regression
"""
y_pred = predict((np.array(6.5).reshape(1, -1))
#predict(6.5) - might work

"""
# Visualising the Regression result
"""
plt.scatter(X, y, color = 'red') #Base results Level Vs Salary from Company
plt.plot(X, regressor.preditct(X), color = 'blue')
plt.title('Truth or Bluff (Regressino Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""
# Visualising the Regression result (For higher resolution of results)
"""
X_grid = np.arange(min(X), max(X), 0.1) #Makes a vector not a matrix, so we restructure below
X_grid = X_grid.reshape((len(X_grid), 1)) #Make a matrix of 0.1 increments 90 columns improve accuracy
plt.scatter(X, y, color = 'red') #Base results Level Vs Salary from Company
plt.plot(X, regressor.preditct(X), color = 'blue')
plt.title('Truth or Bluff (Regressino Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()








