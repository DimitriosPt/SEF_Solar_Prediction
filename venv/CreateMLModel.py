import datetime as dt

import pandas as pd
from CalculateWeatherAttenuation import getPrecipitationChance as PrecipChance
from CalculateWeatherAttenuation import getWeatherData
SONOMALONG= -122.503
SONOMALAT= 38.387
FILEPATH = r"C:\Users\ptdim\Desktop\MLTesting\main_house_garage_cummulative.csv"
df = pd.read_csv(FILEPATH)

df_new = pd.DataFrame()

df_new['Generation [kWh]'] = df["Generation [kWh]"]
df_new['Date & Time'] = df['Date & Time']
# Because ML's cant really use date as a factor as it is ever increasing
# but day of the year is important, I'm breaking up the date into
# year, month, day, and hour to be used as parameters
# although hour will likely become unimportant moving forward
df_new['Year'] = pd.DatetimeIndex(df['Date & Time']).year
df_new['Month'] = pd.DatetimeIndex(df['Date & Time']).month
df_new['Day'] = pd.DatetimeIndex(df['Date & Time']).day
df_new['Hour'] = pd.DatetimeIndex(df['Date & Time']).hour

#maps all the weather attributes forcasted for each date to their respective rows
for index, row in df_new.iterrows():
    date = dt.datetime.strptime(row['Date & Time'], '%m/%d/%Y %H:%M')
    # date = dt.datetime.strptime(row['Date & Time'], '%Y-%m-%d %H:%M:%S')
    daily_forecast = getWeatherData(SONOMALAT, SONOMALONG, date)
    daily_data = daily_forecast.daily.data[0]
    df_new.loc[index, 'Precipitation Intensity'] = daily_data['precipIntensityMax']
    df_new.loc[index, 'Precipitation Probability'] = daily_data['precipProbability']
    df_new.loc[index, 'Dew Point'] = daily_data['dewPoint']
    df_new.loc[index, 'Highest Temp'] = daily_data['temperatureHigh']
    df_new.loc[index, 'Lowest Temp'] = daily_data['temperatureLow']
    df_new.loc[index, 'Humidity'] = daily_data['humidity']
    df_new.loc[index, 'UV Index'] = daily_data['uvIndex']

# The date & Time column that comes from the e-guages is clunky and impossible to graph with
# for most spreadsheet programs, the column at this time is redundant and a reformatted date column
# has been added so this can be safely removed.
df_new = df_new.drop("Date & Time", axis=1)
df_new.to_csv(r"C:\Users\ptdim\Desktop\Stone Edge Farms\Data CSV's\main_house_garageML.csv", index=None)
print(df_new.head())
