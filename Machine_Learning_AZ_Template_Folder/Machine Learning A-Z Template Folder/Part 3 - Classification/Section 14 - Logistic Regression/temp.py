# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 08:41:24 2017

@author: Sonam
"""
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Creating the data set
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:, [2,3]].values
y=dataset.iloc[:,4].values

#Splitting the data set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Fitting the logistics regression data
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#Predicting the value of the test set
y_pred=classifier.predict(X_test)

#Making the comfusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Visualizing the test set results