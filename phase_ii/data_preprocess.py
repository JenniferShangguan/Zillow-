# pre process data and save them in data/features.csv

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from functools import reduce

print("Reading house_price_median.csv")
price = pd.read_csv("data/house_price_median.csv",
                    names=["County", "Price"],
                    header=None)

# commute
print("Reading commute.csv")
df = pd.read_csv("data/commute.csv", header=0)
rsum = df.groupby(['RCounty'])['Number'].sum().reset_index()
wsum = df.groupby(['WCounty'])['Number'].sum().reset_index()
df = pd.merge(wsum,
              rsum,
              left_on='WCounty',
              right_on='RCounty',
              suffixes=("_w", "_r"),
              sort=False)
commute = pd.DataFrame(data={
    'County': df.WCounty,
    'Job Opportunity': df.Number_w / df.Number_r
})

# Income
print("Reading income.csv")
df = pd.read_csv("data/income.csv", header=0)
income = df.groupby(['County'])['Median Household Income'].mean().reset_index()

# unemployment
print("Reading unemployment.csv")
df = pd.read_csv("data/unemployment.csv", header=0)
for i in range(0, len(df['Month of Date'])):
    # date = df['Month of Date'][i]
    date = df.at[i, 'Month of Date']
    try:
        df.at[i, 'Month of Date'] = datetime.strptime(date, '%b-%y').year
    except:
        df.at[i, 'Month of Date'] = datetime.strptime(date, '%B %Y').year
df.rename(columns={'Area Name': 'County',
                   'Accurate rate': 'Unemployment Rate', 'Month of Date': 'Year'}, inplace=True)
unemployment = df.loc[df["Year"] > 2005].groupby(
    ['County'])['Unemployment Rate'].mean().reset_index()

# migration
print("Reading migration.csv")
df = pd.read_csv("data/migration.csv", header=0)
df.County = df.County + ' County'
a = np.asarray(df.loc[df["Year"] == "2009-2013"].total) + 1
b = np.asarray(df.loc[df["Year"] == "2013-2017"].total) + 1
migr = (a * b - 1).tolist()
migration = pd.DataFrame(data={
    'County': df.loc[df["Year"] == "2009-2013"].County,
    'Migration Rate': migr
})

# Public Transport
print("Reading geography.csv")
geo = pd.read_csv("data/geography.csv", header=0)
print("Reading publictransport.csv")
df = pd.read_csv("data/publictransport.csv", header=0)
df.County = df.County + ' County'
df = df.merge(geo, on='County', sort=False)
df["Public Transportation Access"] = (
    df.Caltrain + df.Bart) / df.Land_suare_miles
publictransport = df[["County", "Public Transportation Access"]]

# Merge everything to one df
data_frames = [commute, income, unemployment,
               migration, publictransport, price]
df_merged = reduce(lambda left, right: pd.merge(left, right, on=['County'],
                                                how='outer'), data_frames)

## Output ##
df_merged.to_csv('data/features.csv', index=False)
print("Data processing finished, generated data/features.csv")
