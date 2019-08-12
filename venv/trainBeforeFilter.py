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

DATASET_LOCATION = r"C:\Users\ptdim\Desktop\Stone Edge Farms\Data CSV's\agShedML.csv"

data = pd.read_csv(DATASET_LOCATION)
feature_cols = ['Generation [kWh]', 'Day', 'Month',
                'Precipitation Intensity', 'Precipitation Probability', 'Dew Point',
                'Highest Temp', 'Lowest Temp', 'Humidity', 'UV Index']

data.drop(columns="Hour", inplace=True)
# The data that we pull from the egauges are cummulative, so if we want to get an actual daily solar production reading
# we have to subtract yesterday's totals from today's totals to see how much power we generated on the previous day
data["Generation [kWh]"] = data["Generation [kWh]"].diff(periods=-1)

# because getting the daily readings is obtained by subtracting one value from the one below it, this means the bottom
# row of data will either be massive (and incorrect), or result in NaN, so we are just dropping the bottom row to avoid
# these issues
data.drop(data.tail(1).index,inplace=True)

# Filters the Generation column with a Butterworth Filter to make it less noisy. Training the model
# off of unfiltered data led to the model predicting massive osscilations due to the inconsistancy of
# the data we were pulling in.
a,b = butter(3, 0.05)

X_train, X_test = train_test_split(data, test_size=0.3)

# Sets the dependant variables into their own data structures
y_train = X_train["Generation [kWh]"]
y_test = X_test["Generation [kWh]"]
# Removes the dependant variables from the X sets
X_train = X_train.drop(columns="Generation [kWh]")
X_test = X_test.drop(columns="Generation [kWh]")

ridge = Ridge(alpha=.005, normalize=True).fit(X_train,y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

filtered_data = data.copy()
filtered_data["Predicted Generation [kWh]"] = filter(a,b,filtered_data["Generation [kWh]"])

prediction_list = []
for index, row in data.iterrows():
    year = int(row["Year"])
    month = int(row["Month"])
    day = int(row["Day"])
    day_to_predict = datetime.date(year=year, month=month, day=day)
    precipitation_intensity = row["Precipitation Intensity"]
    precipitation_probability = row["Precipitation Probability"]
    dew_point = row["Dew Point"]
    highest_temp = row["Highest Temp"]
    lowest_temp = row["Lowest Temp"]
    humidity = row['Humidity']
    uv_index = row['UV Index']

    prediction_values = numpy.array([[year, month, day,
                                      precipitation_intensity,
                                      precipitation_probability,
                                      dew_point, highest_temp,
                                      lowest_temp, humidity, uv_index]])

    prediction = Ridge.predict(ridge, prediction_values)
    write_string = str(prediction).strip("[]")
    prediction_list.append(prediction)

    #Predictions are outputted with brackets around the number which is a nuisance when
    # trying to graph the data in excel so I remove them before writing to the csv
    filtered_data.loc[index, "Predicted Generation [kWh]"] = str(prediction).strip("[]")

    filtered_data.loc[index, "Date"] = day_to_predict
    write_string = write_string.strip("[]")

# applies the filter to our graphs to remove the noise
filtered_predictions = filter(a, b, prediction_list, axis=0)
filtered_actual = filter(a, b, data["Generation [kWh]"], axis=0)
filtered_data["Filtered Predictions"] = filtered_predictions
filtered_data["Filtered Actual"] = filtered_actual

print('------------------------')
print("P Values")
print('------------------------')

p_vals = stats.coef_pval(ridge, X_train, y_train)

columns = []
i = 0
for column in data.columns:
    columns.append(column)

columns.remove("Generation [kWh]")

i = 0
p_vals = p_vals[0:10]
for column in columns:
    print(f'{column}: {p_vals[i]}')
    i += 1

#provides a simple graph of the filtered data so the user can see if its something worth saving
plt.plot(filtered_actual, 'g')
plt.plot(filtered_predictions, 'y')
plt.show()


print(f' \nThe Predicted Power Generation for tomorrow is: {make_prediction(ridge)} \n')

if(input("Would you like to save this csv? (y/n): ").upper() == 'Y'):
    file_name = input("What would you like to name the file?: ")
    file_name = file_name + ".csv"
    save_directory = r"C:\Users\ptdim\Desktop\MLTesting\\"
    save_location = save_directory + file_name
    filtered_data.to_csv(save_location, index=None)
