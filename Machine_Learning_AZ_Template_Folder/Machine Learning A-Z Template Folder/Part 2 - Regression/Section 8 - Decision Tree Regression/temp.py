# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 05:49:57 2017

@author: Sonam
"""
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading the CSV file and creating the data variables
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:,2].values

#Create the decission tree regressor
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Predict the value of regressor
y_pred=regressor.predict(6.5)

#Plot the decision tree regressor
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth of Bluff(Decision Tree)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
