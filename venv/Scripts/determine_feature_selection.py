#TODO figure out why the actual data is being recorded as already filtered 8/5/2019

# Dimitrios Papageorgacopoulos
# Wooster Engineering
# Summer 2019
# Ridge Regression predictive model for solar power generation
# dimitrios@sefmicrogrid.com

# This program will take a csv of the format
# [Generation, Year, Day, Month, Hour, Precipitation Intensity, Precipitation Chance, Dew Point,
# Highest Temp, Lowest temp, Humidity, UV Index]
# Where day is the numbered day of the month and month is the number of the month, hour is the time
# at which the data is collected in military time. Working to remove the hour parameter.
# The program takes this data and uses machine learning to create a ridge regression model.
# Once the model is built, predictions can be made using the
# Ridge.predict(ridge, data_test) method with data_test taking the form of
# a nested numpy array such that testing March 30 2020 with a forecasted precipitation chance
# of 45% would look like [[2020, 3, 30, 0, 0.45]]. To make a prediction, you have to use the make_prediction() function
# which will take a day to predict as a parameter.
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


def make_prediction(ml_model):
    date = datetime.datetime.today() + datetime.timedelta(hours=24)
    year= date.year
    month = date.month
    day = date.day

    #epoch_time = math.floor(date.timestamp())  # converts to Epoch
    SONOMALONG = -122.503
    SONOMALAT = 38.387
    forecast = getWeatherData(SONOMALAT, SONOMALONG, date)
    daily_forecast = forecast.daily.data[0]

    # the 4th parameter is just a 0 here since that represents the hour of pulling the data
    # and it shouldnt affect output of any kind
    prediction_parameters = numpy.array([[year, day, month, daily_forecast['precipIntensityMax'],
                                          daily_forecast['precipProbability'], daily_forecast['dewPoint'],
                                          daily_forecast['temperatureHigh'], daily_forecast['temperatureLow'],
                                          daily_forecast['humidity'], daily_forecast['uvIndex']]])

    return Ridge.predict(ml_model, prediction_parameters)

dataset_location = r"C:\Users\ptdim\Desktop\Stone Edge Farms\Data CSV's\main_house_garageML.csv"
butler_data_path= r"C:\Users\ptdim\Desktop\MLTesting\butlerYearly.csv"
butler_data = pd.read_csv(butler_data_path)
data = pd.read_csv(dataset_location)
feature_cols = ['Generation [kWh]', 'Day', 'Month',
                'Precipitation Intensity', 'Precipitation Probability', 'Dew Point',
                'Highest Temp', 'Lowest Temp', 'Humidity', 'UV Index']

data.drop(columns="Hour", inplace=True)
# The data that we pull from the egauges are cummulative, so if we want to get an actual daily solar production reading
# we have to subtract yesterday's totals from today's totals to see how much power we generated on the previous day
data["Generation [kWh]"] = data["Generation [kWh]"].diff(periods=-1)

butler_data["Generation [kWh]"] = butler_data["Generation [kWh]"].diff(periods=-1)

# because getting the daily readings is obtained by subtracting one value from the one below it, this means the bottom
# row of data will either be massive (and incorrect), or result in NaN, so we are just dropping the bottom row to avoid
# these issues
data.drop(data.tail(1).index,inplace=True)
butler_data.drop(data.tail(1).index,inplace=True)

# Filters the Generation column with a Butterworth Filter to make it less noisy. Training the model
# off of unfiltered data led to the model predicting massive osscilations due to the inconsistancy of
# the data we were pulling in.
a,b = butter(3, 0.05)

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
plt.savefig("Barplot.png")
plt.show()
