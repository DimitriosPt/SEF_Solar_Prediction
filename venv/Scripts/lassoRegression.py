#TODO figure out why the actual data is being recorded as already filtered 8/5/2019

# Dimitrios Papageorgacopoulos
# Wooster Engineering
# Summer 2019
# Ridge Regression predictive model for solar power generation
# dimitrios@sefmicrogrid.com

# This program will take a csv of the format [Generation, Day, Month, Hour, Precipitation Chance]
# Where day is the numbered day of the month and month is the number of the month, hour is the time
# at which the data is collected in military time. Working to remove the hour parameter.
# The program takes this data and uses machine learning to create a ridge regression model.
# Once the model is built, predictions can be made using the
# Ridge.predict(ridge, data_test) method with data_test taking the form of
# a nested numpy array such that testing March 30 2020 with a forecasted precipitation chance
# of 45% would look like [[2020, 3, 30, 0, 0.45]]

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd
import datetime
import csv
import random
import numpy
from scipy.signal import butter as butter
from scipy.signal import filtfilt as filter


dataset_location = r"C:\Users\ptdim\Desktop\Stone Edge Farms\Data CSV's\main_house_garageML.csv"
butler_data_path= r"C:\Users\ptdim\Desktop\MLTesting\butlerYearly.csv"
butler_data = pd.read_csv(butler_data_path)
data = pd.read_csv(dataset_location)
feature_cols = ['Generation [kWh]', 'Day', 'Month',
                'Precipitation Intensity', 'Precipitation Probability', 'Dew Point',
                'Highest Temp', 'Lowest Temp', 'Humidity', 'UV Index']

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

X_train, X_test = train_test_split(data, test_size=0.3)

# Sets the dependant variables into their own data structures
y_train = X_train["Generation [kWh]"]
y_test = X_test["Generation [kWh]"]
# Removes the dependant variables from the X sets
X_train = X_train.drop(columns="Generation [kWh]")
X_test = X_test.drop(columns="Generation [kWh]")

lasso = Lasso(alpha=0.1)
lasso.fit(X_train,X_test)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))

prediction_list = []

filtered_data = data.copy()
filtered_data["Predicted Generation [kWh]"] = filter(a,b,filtered_data["Generation [kWh]"])

for index, row in data.iterrows():

    year = int(row["Year"])
    month = int(row["Month"])
    day = int(row["Day"])
    hour = int(row["Hour"])
    day_to_predict = datetime.date(year=year, month=month, day=day)
    precipitation_intensity = row["Precipitation Intensity"]
    precipitation_probability = row["Precipitation Probability"]
    dew_point = row["Dew Point"]
    highest_temp = row["Highest Temp"]
    lowest_temp = row["Lowest Temp"]
    humidity = row['Humidity']
    uv_index = row['UV Index']

    prediction_values = numpy.array([[year, month, day, hour, precipitation_intensity,
                                      precipitation_probability ,dew_point, highest_temp, lowest_temp,
                                      humidity, uv_index]])

    prediction = Lasso.predict(lasso, prediction_values)
    write_string = str(prediction).strip("[]")
    prediction_list.append(prediction)

    #Predictions are outputted with brackets around the number which is a nuisance when
    # trying to graph the data in excel so I remove them before writing to the csv
    filtered_data.loc[index, "Predicted Generation [kWh]"] = str(prediction).strip("[]")

    filtered_data.loc[index, "Date"] = day_to_predict
    write_string = write_string.strip("[]")
    # writer.writerow([day_to_predict, write_string])





filtered_predictions = filter(a, b, prediction_list, axis=0)
filtered_actual = filter(a, b, data["Generation [kWh]"], axis=0)
#filtered_data["Predicted Generation"] = prediction_list
#filtered_data["Filtered Predictions"] = filtered_predictions
#filtered_data["Filtered Actual"] = filtered_actual

#plt.plot(data["Generation [kWh]"], 'b')
plt.plot(filtered_actual, 'g')
#plt.plot(prediction_list, 'r')
plt.plot(filtered_predictions, 'y')
plt.show()

if(input("Would you like to save this csv? (y/n): ").upper() == 'Y'):
    file_name = input("What would you like to name the file?: ")
    file_name = file_name + ".csv"
    save_directory = r"C:\Users\ptdim\Desktop\MLTesting\\"
    save_location = save_directory + file_name
    filtered_data.to_csv(save_location, index=None)

#plt.plot(filtered_predictions)

# today = numpy.array( [[2019, 7, 20, 0, 0.01]])
# yesterday = numpy.array( [[2019, 7,19 , 0, 0.01]])
# print(f'For just today: {Ridge.predict(ridge, today) - Ridge.predict(ridge, yesterday)}')
#
