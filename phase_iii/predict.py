#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:04:20 2019

@author: yanzhou
"""
import pandas as pd
import numpy 
#from sklearn import linear_model
import sklearn

df = pd.read_csv('summary_noempty.csv')
X = df[['Unemployment','Income','Rate']]
Y = df[['Price']]
W = pd.read_csv('test.csv')
W = W.drop(columns=['Region','Date','Price'])
lm = sklearn.linear_model.LinearRegression()
model = lm.fit(X,Y)
predictions = lm.predict(W)
#print(predictions)
numpy.savetxt("Price.csv", predictions, delimiter=",")
