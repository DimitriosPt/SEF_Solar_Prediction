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

from sklearn.linear_model import Ridge
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


dataset_location = r"C:\Users\ptdim\Desktop\Stone Edge Farms\Data CSV's\butlerML.csv"
butler_data_path= r"C:\Users\ptdim\Desktop\MLTesting\butlerYearly.csv"
butler_data = pd.read_csv(butler_data_path)

data = pd.read_csv(dataset_location)
feature_cols = ['Generation [kWh]', 'Day', 'Month',
                'Precipitation Intensity', 'Precipitation Probability', 'Dew Point',
                'Highest Temp', 'Lowest Temp', 'Humidity', 'UV Index']
X_train, X_test = train_test_split(data, test_size=0.2)

actual_generation = data["Generation [kWh]"]
# Sets the dependant variables into their own data structures
y_train = X_train["Generation [kWh]"]
y_test = X_test["Generation [kWh]"]
# Removes the dependant variables from the X sets
X_train = X_train.drop(columns="Generation [kWh]")
X_test = X_test.drop(columns="Generation [kWh]")

ridge = Ridge().fit(X_train,y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
prediction_list = []
df = pd.DataFrame()
with open('predictions.csv', 'w', newline='') as prediction_file:
    writer = csv.writer(prediction_file)
    writer.writerow(['Date', 'Predicted Output [kWh]', 'Filtered Data'])

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

        prediction = Ridge.predict(ridge, prediction_values)
        write_string = str(prediction)
        prediction_list.append(prediction)
        #Predictions are outputted with brackets around the number which is a nuisance when
        # trying to graph the data in excel so I remove them before writing to the csv
        data.loc[index, "Predicted Generation [kWh]"] = str(prediction).strip("[]")
        write_string = write_string.strip("[]")
        writer.writerow([day_to_predict, write_string])


a, b = butter(3, 0.05)
filtered_predictions = filter(a, b, prediction_list, axis=0)
filtered_actual = filter(a, b, butler_data["Generation [kWh]"], axis=0)
data["Filtered Predictions"] = filtered_predictions
data["Actual Generation [kWh]"] = butler_data["Generation [kWh]"]
data["Filtered Actual"] = filtered_actual


data.to_csv(r"C:\Users\ptdim\Desktop\MLTesting\filteredCSV.csv")
plt.plot(filtered_predictions)

plt.show()
# today = numpy.array( [[2019, 7, 20, 0, 0.01]])
# yesterday = numpy.array( [[2019, 7,19 , 0, 0.01]])
# print(f'For just today: {Ridge.predict(ridge, today) - Ridge.predict(ridge, yesterday)}')
#
