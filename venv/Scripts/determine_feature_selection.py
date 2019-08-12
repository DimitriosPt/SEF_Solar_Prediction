# Dimitrios Papageorgacopoulos
# Wooster Engineering
# Summer 2019
# Ridge Regression predictive model for solar power generation
# dimitrios@sefmicrogrid.com

import math
import datetime
import csv
import random
import numpy
import seaborn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from regressors import stats
import pandas as pd
from scipy.signal import butter as butter
from scipy.signal import filtfilt as filter
from CalculateWeatherAttenuation import getWeatherData


DATA_FILE_PATH = r"C:\Users\ptdim\Desktop\Stone Edge Farms\Data CSV's\agShedML.csv"
data = pd.read_csv(DATA_FILE_PATH)
feature_cols = ['Generation [kWh]', 'Day', 'Month',
                'Precipitation Intensity', 'Precipitation Probability', 'Dew Point',
                'Highest Temp', 'Lowest Temp', 'Humidity', 'UV Index']

data.drop(columns="Day", inplace=True)
data.drop(columns="Hour", inplace=True)
data.drop(columns="Month", inplace=True)
data.drop(columns="Year", inplace=True)

# The data that we pull from the egauges are cummulative, so if we want to get an actual daily solar production reading
# we have to subtract yesterday's totals from today's totals to see how much power we generated on the previous day
data["Generation [kWh]"] = data["Generation [kWh]"].diff(periods=-1)

# because getting the daily readings is obtained by subtracting one value from the one below it, this means the bottom
# row of data will either be massive (and incorrect), or result in NaN, so we are just dropping the bottom row to avoid
# these issues
data.drop(data.tail(1).index,inplace=True)

sum_values = [0] * 10
for i in range (0, 1000):

    X_train, X_test = train_test_split(data, test_size=0.3)

    # Sets the dependant variables into their own data structures
    y_train = X_train["Generation [kWh]"]
    y_test = X_test["Generation [kWh]"]
    # Removes the dependant variables from the X sets
    X_train = X_train.drop(columns="Generation [kWh]")
    X_test = X_test.drop(columns="Generation [kWh]")

    ridge = Ridge(alpha=.005, normalize=True).fit(X_train,y_train)

    p_vals = stats.coef_pval(ridge, X_train, y_train)

    columns = []
    i = 0
    for column in X_train.columns:
        columns.append(column)

    i = 0
    p_vals = p_vals[0:10]
    for value in p_vals:
        #print(f'{column}: {p_vals[i]}')
        sum_values[i] += p_vals[i]
        i += 1

for i in range(0, len(sum_values)):
    sum_values[i] = sum_values[i] / 1000
# Plots a bar graph showing the coefficients
plt.figure(dpi=300)
barplot_df = pd.DataFrame(list(zip(columns, sum_values)), columns= ["Factor", "P Value"])
barplot_df = barplot_df.sort_values('P Value')
seaborn.barplot(x="P Value", y="Factor", data=barplot_df, palette="coolwarm")
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("featureBarplot.svg")
plt.show()
