import pandas as pd
import datetime as dt
from CalculateWeatherAttenuation import calculateWeatherAttenuation as PrecipChance
SONOMALONG=-122.503
SONOMALAT=38.387
FILEPATH=r"C:\Users\ptdim\Desktop\Stone Edge Farms\yearLongDailyJuly18.csv"
df = pd.read_csv(FILEPATH)

df_new = pd.DataFrame()

df_new['Generation [kWh]'] = df["Generation [kWh]"]
df_new['Date & Time'] = df['Date & Time']
# Because ML's cant really use date as a factor as it is ever increasing
# but day of the year is important, I'm breaking up the date into
# year, month, day, and hour to be used as parameters
df_new['Year'] = pd.DatetimeIndex(df['Date & Time']).year
df_new['Month'] = pd.DatetimeIndex(df['Date & Time']).month
df_new['Day'] = pd.DatetimeIndex(df['Date & Time']).day
df_new['Hour'] = pd.DatetimeIndex(df['Date & Time']).hour
for index, row in df_new.iterrows():
    date = dt.datetime.strptime(row['Date & Time'], '%Y-%m-%d %H:%M:%S')
    #df_new.loc[index, 'Precipitation Chance'] = PrecipChance(SONOMALAT, SONOMALONG, date)

df_new = df_new.drop("Date & Time", axis=1)
df_new.to_csv(r"C:\Users\ptdim\Desktop\Stone Edge Farms\Data CSV's\mlData.csv", index=None)
print(df_new.head())