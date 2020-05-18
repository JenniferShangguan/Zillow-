# plot heatmap by reading data from data/features.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

sns.set(style="darkgrid")


df = pd.read_csv("data/features.csv")
y = df['Price'].values.reshape(-1, 1) * 1e-3
x_cols = df.columns.difference(['Price', 'County'])
county = df['County']
# print(x_cols)
df_error = pd.DataFrame(index=county, columns=x_cols)
mean_error = {}
for col in x_cols:
    x = df[col].values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    yp = reg.predict(x)
    error = np.abs(y - yp)
    print(col, ", L1 Residual: ", np.mean(error), " Coef: ", reg.coef_)
    mean_error[col] = np.mean(error)
    df_error[col] = error
row = pd.Series(mean_error, name='Mean')
df_error = df_error.append(row)
X = df[x_cols]
reg = LinearRegression().fit(X, y)
yp = reg.predict(X)
error = np.abs(y - yp)
print("All Features: ", "L1 Residual: ", np.mean(error))
sns.set(font_scale=1.8)
ax = plt.subplot(111)
plt.subplots_adjust(left=0.2, top=0.9, bottom=0.2)
sns.heatmap(df_error, cmap='Oranges', annot=True, fmt="0.1f", ax=ax)
ax.set_title("Linear Regression L1 Residual (unit $1000)")
plt.xticks(rotation=10)
fig = plt.gcf()
fig.set_size_inches(18, 10.5)
plt.show()
