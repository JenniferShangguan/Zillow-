# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
df = pd.read_csv('unemployment.csv')
df.index = pd.to_datetime(df['Time'],format="%b-%y")
for column in df['Area Name'].unique():
    train = df.loc[df['Area Name'] == column ]
    train = train.drop(columns=['Time','Area Name'])
    
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.predict(len(train), len(train)+27, dynamic=True)
    print(column)
    print(forecast)

