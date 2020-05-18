# This script plot the linear regression result by reading data/features.csv
# it plots the linear regression result for
# Meadian Household income vs Meadian House Price
# also plots the feature importance when using all
# features

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale

df = pd.read_csv("data/features.csv", header=0)

x_col = df.columns.difference(['Price', 'County'])

# Plot Income
# Linear Regression
xt = np.array(df["Median Household Income"]).reshape(-1, 1) / 1000
yt = np.array(df.Price).reshape(-1, 1) / 1000
regr = linear_model.LinearRegression()
regr.fit(xt, yt)
xp = np.arange(np.min(xt), np.max(xt), 0.1).reshape(-1, 1)
yp = regr.predict(xp)
# plot line
plt.plot(xp, yp)
# plot Data Point
x = df["Median Household Income"] / 1000
y = df["Price"] / 1000
plt.scatter(x, y, marker='o', color='red')
for i, county_name in enumerate(df.County):
    plt.text(x[i], y[i], county_name, fontsize=18)

# set up labels
ax = plt.gca()
ax.tick_params(axis="x", labelsize=18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%d k'))
ax.tick_params(axis="y", labelsize=18)
ax.yaxis.set_major_formatter(FormatStrFormatter('%d k'))
plt.xlabel('Average Median Household Income from 2006 to 2017', fontsize=18)
plt.ylabel('Median House Price', fontsize=18)
plt.title(
    'Linear Regression of Median Household Income and Median House Price',
    fontsize=24)
plt.get_current_fig_manager().full_screen_toggle()

plt.figure()
# Plot Feature Importance by Linear Regression
# LinearRegression normalized
y = minmax_scale(np.array(df.Price).reshape(-1, 1))
X = minmax_scale(np.array(df[x_col]))
reg = LinearRegression().fit(X, y)
yp = reg.predict(X)
# plot lines
for i in range(0, len(x_col)):
    x = np.arange(0, 1, 0.1)
    abs_coef = abs(reg.coef_[0][i])
    y = abs_coef * x + reg.intercept_[0]
    label = x_col[i] + ' , Coeff: ' + str(round(reg.coef_[0][i], 2))
    plt.plot(x, y, label=label)
    plt.legend(prop={'size': 20})
plt.gca().tick_params(axis="x", labelsize=18)
plt.gca().tick_params(axis="y", labelsize=18)
plt.xlabel('Features (Normalized)', fontsize=24)
plt.ylabel('Median House Price (Normalized)', fontsize=24)
plt.title('Feature Importance(abs) by Linear Regression', fontsize=30)
plt.get_current_fig_manager().full_screen_toggle()
plt.show()
