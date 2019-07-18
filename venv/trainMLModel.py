# Dimitrios Papageorgacopoulos
# Wooster Engineering
# Summer 2019
# Ridge Regression predictive model for solar power generation

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
from matplotlib import pyplot as plt
import pandas as pd
import datetime
import csv
import random
import numpy

dataset_location = r"C:\Users\ptdim\Desktop\Stone Edge Farms\Data CSV's\mlData.csv"
data = pd.read_csv(dataset_location)
feature_cols = ['Generation [kWh]', 'Day', 'Month', 'Precipitation Chance']
X_train, X_test = train_test_split(data, test_size=0.2)

# Sets the dependant variables into their own data structures
y_train = X_train["Generation [kWh]"]
y_test = X_test["Generation [kWh]"]
# Removes the dependant variables from the X sets
X_train = X_train.drop(columns="Generation [kWh]")
X_test = X_test.drop(columns="Generation [kWh]")

# performs a ridge regression to determine relationship between
# independent and dependent variables
ridge = Ridge().fit(X_train, y_train)

# print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

with open('predictions.csv', 'w', newline='') as prediction_file:
    writer = csv.writer(prediction_file)
    writer.writerow(['Date', 'Predicted Output [kWh]'])
    for index, row in data.iterrows():
        year = int(row["Year"])
        month = int(row["Month"])
        day = int(row["Day"])
        hour = int(row["Hour"])
        day_to_predict = datetime.date(year=year, month=month, day=day)
        precipitation_chance = row["Precipitation Chance"]

        prediction_values = numpy.array([[year, month, day, hour, precipitation_chance]])
        prediction = Ridge.predict(ridge, prediction_values)
        print(f'Date: {day_to_predict} Predicted Generation: {prediction}')
        write_string = str(prediction)
        write_string = write_string.strip("[]")
        writer.writerow([day_to_predict, write_string])

print(f'For just today: {Ridge.predict(ridge, date_to_predict2) - Ridge.predict(ridge, date_to_predict)}')

