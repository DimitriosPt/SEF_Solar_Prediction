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
import pandas as pd

today = datetime.datetime.now()
dst_start = datetime.datetime(year=today.year, month=3, day=10)
dst_end = datetime.datetime(year=today.year, month=11, day=3)
today = pytz.timezone('US/Pacific').localize(today)
sonomaLatitude = 38.387
sonoma_lat_radians = math.radians(sonomaLatitude)
sonomaLongitude = -122.503
epoch_time = today.timestamp()
angle_between_sun_panel = angle(sonoma_lat_radians, today)
atmospheric_attenuation = atmospheric_atten(sonoma_lat_radians, today)
# precip_chance = get_weather_atten(sonomaLatitude, sonomaLongitude, today)

# TODO figure out why angle attenuation is slightly offset of
# atmospheric attenuation

# TODO fix the daylight savings problem
# maybe try brute forcing it and making it so if the day is between march 10 and nov 3 add an hour?


time_format = '%d/%m/%Y %H:%M'
start_date = today - datetime.timedelta(days=5)
with open('energyForecast.csv', 'w', newline='') as energyFile:
    writer = csv.writer(energyFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Date', 'Attenuation Due To Angle', 'Atmospheric Attenuation',
                     # 'Precipitation Chance',
                     'Predicted Energy Output'])
    pacific_tz = pytz.timezone('US/Pacific')
    while start_date <= today:
        angle_between_sun_panel = angle(sonoma_lat_radians, start_date)
        atmospheric_attenuation = atmospheric_atten(sonoma_lat_radians, start_date)
        # precip_chance = get_weather_atten(sonomaLatitude, sonomaLongitude, start_date)
        energy = angle_between_sun_panel * atmospheric_attenuation \
                 * math.cos((1 - math.radians(precip_chance)))
        date_to_write = start_date

        if pacific_tz.localize(dst_start) <= date_to_write <= pacific_tz.localize(dst_end):
            date_to_write = date_to_write + datetime.timedelta(hours=2)
        writer.writerow([date_to_write.strftime(time_format), angle_between_sun_panel, atmospheric_attenuation,
                         # precip_chance,
                         energy])
        print(start_date)
        start_date = start_date + datetime.timedelta(minutes=15)
