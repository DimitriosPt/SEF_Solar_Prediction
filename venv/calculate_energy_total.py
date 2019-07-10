from CalculateAngleAttenuation import calculate_angle_between_panel_sun as angle
from CalculateAtmosphericAttenuation import calculate_atmospheric_attenuation as atmospheric_atten
from CalculateWeatherAttenuation import calculateWeatherAttenuation as get_weather_atten
import datetime
import pytz
import math
import csv
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import pandas

today = datetime.datetime.now()
today = pytz.timezone('US/Pacific').localize(today)
sonomaLatitude = 38.387
sonoma_lat_radians = math.radians(sonomaLatitude)
sonomaLongitude = -122.503
epoch_time = today.timestamp()
angle_between_sun_panel = angle(sonoma_lat_radians, today)
atmospheric_attenuation = atmospheric_atten(sonoma_lat_radians, today)
precip_chance = get_weather_atten(sonomaLatitude, sonomaLongitude, today)

# TODO see if I really need to take cos of precipitation chance
# TODO graph over course of a day to see if sin wave emerges

time_format = '%d/%m/%Y %H:%M'
start_date = today - datetime.timedelta(days=5)
with open('energyForecast.csv', 'w') as energyFile:
      writer = csv.writer(energyFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['Date', 'Angle Between Sun and Panels', 'Atmospheric Attenuation',
                        #'Precipitation Chance',
                        'Predicted Energy Output'])
      while start_date <= today:
            angle_between_sun_panel = angle(sonoma_lat_radians, start_date)
            atmospheric_attenuation = atmospheric_atten(sonoma_lat_radians, start_date)
            #precip_chance = get_weather_atten(sonomaLatitude, sonomaLongitude, start_date)
            energy = 1000 * angle_between_sun_panel * atmospheric_attenuation * math.cos((1 - precip_chance))
            writer.writerow([start_date.strftime(time_format), angle_between_sun_panel, atmospheric_attenuation,
                              precip_chance, energy])

            start_date = start_date + datetime.timedelta(minutes=15)

style.use('ggplot')
per_data=np.genfromtxt('energyForecast.csv',usecols=(0,3),delimiter=',',skip_header=1)
plt.title('Energy Forecast')
plt.xlabel('Date Time')
plt.ylabel('Energy kW')
plt.grid
plt.show()