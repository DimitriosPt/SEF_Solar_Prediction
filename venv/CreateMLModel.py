import datetime as dt
import os, sys
import pandas as pd
from CalculateWeatherAttenuation import getPrecipitationChance as PrecipChance
from CalculateWeatherAttenuation import getWeatherData

# Print iterations progress
# This function was found here, I did not write this
# https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

SONOMALONG=-122.503
SONOMALAT=38.387
FILEPATH=r"C:\Users\ptdim\Desktop\MLTesting\agshedClean.csv"
FILEPATH_OUT=r"C:\Users\ptdim\Desktop\Stone Edge Farms\Data CSV's\agShedML.csv"

try:
    df = pd.read_csv(FILEPATH)

except:
    print("File location not valid, exiting")
    exit()

df_new = pd.DataFrame()

df_new['Generation [kWh]'] = df['Generation [kWh]']
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
i = 0
for index, row in df_new.iterrows():

    clear = lambda: os.system('cls')
    clear()
    print_progress(i, df_new.shape[0])

    try:
        date = dt.datetime.strptime(row['Date & Time'], '%Y-%m-%d %H:%M:%S')
    except:
        print("There was an error interpretting the date format in the file \n "
              "This can happen if you used daily averages instead of cummulative data from the e-gauge")

    # date = dt.datetime.strptime(row['Date & Time'], '%Y-%m-%d %H:%M:%S')
    try:
         daily_forecast = getWeatherData(SONOMALAT, SONOMALONG, date)
    except:
        print("Error gathering forecast data. If this persists change the API key in calculateWeatherAttenuation.py as "
              "there is a limit of 1000 calls per day")

    daily_data = daily_forecast.daily.data[0]
    df_new.loc[index, 'Precipitation Intensity'] = daily_data['precipIntensityMax']
    df_new.loc[index, 'Precipitation Probability'] = daily_data['precipProbability']
    df_new.loc[index, 'Dew Point'] = daily_data['dewPoint']
    df_new.loc[index, 'Highest Temp'] = daily_data['temperatureHigh']
    df_new.loc[index, 'Lowest Temp'] = daily_data['temperatureLow']
    df_new.loc[index, 'Humidity'] = daily_data['humidity']
    df_new.loc[index, 'UV Index'] = daily_data['uvIndex']
    i += 1

# The date & Time column that comes from the e-guages is clunky and impossible to graph with
# for most spreadsheet programs, the column at this time is redundant and a reformatted date column
# has been added so this can be safely removed.
df_new = df_new.drop("Date & Time", axis=1)
print("\n")
print(df_new.head())
df_new.to_csv(FILEPATH_OUT, index=None)

