# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
df = pd.read_csv('interest_Rate.csv')
df.index = pd.to_datetime(df['Time'],format="%d-%b-%y")
df = df.drop(columns=['Time'])

model = SARIMAX(df, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)

forecast = model_fit.predict(len(df), len(df)+25, dynamic=True)
print(forecast)
