#Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Linear Regression model
from sklearn.linear_model import  LinearRegression
linear_regressor= LinearRegression()
linear_regressor.fit(X,y)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)
linear_regression2=LinearRegression()
linear_regression2.fit(X_poly,y)


#Visualizing the linear regression model
plt.scatter(X,y,color='blue')
plt.plot(X,linear_regressor.predict(X),color='red')
plt.title('Polynomial Linear RegressionModel(Linear)')
plt.xlabel('Level')
plt.ylabel('Salary')

#Visualizing the polynomial regression
plt.scatter(X,y,color='blue')
plt.plot(X,linear_regression2.predict(X_poly),color='red')
plt.title('Polynomial Linear RegressionModel(Polynomial)')
plt.xlabel('Level')
plt.ylabel('Salary')

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='blue')
plt.plot(X_grid,linear_regression2.predict(poly.fit_transform(X_grid)),color='red')
plt.title('Polynomial Linear RegressionModel(Polynomial)')
plt.xlabel('Level')
plt.ylabel('Salary')

#Predicting with the linear regression
y_pred1=linear_regressor.predict(6.5)

#Predicting with the polynomial regression
y_pred2=linear_regression2.predict(poly.fit_transform(6.5))