# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
df = pd.read_csv('income.csv')
df.index = pd.to_datetime(df['Time'],format="%Y")
for column in df['County'].unique():
    train = df.loc[df['County'] == column ]
    train = train.drop(columns=['Time','County'])
    
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False, enforce_invertibility=False)
    
    forecast = model_fit.predict(len(train), len(train)+7)
    print(column)
    print(forecast)