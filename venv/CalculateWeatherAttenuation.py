# Dimitrios Papageorgacopoulos
# Intern at Stone Edge Farms Microgrid Project
# June 21 2019

# this script uses the Dark Sky api
# documentation for the wrapper be found here https://pypi.org/project/darkskylib/
# documentation for the dark sky api is found here : https://darksky.net/dev/docs

# IMPORTANT
# If this code is ever used to publically display data we must include
# "Powered By Dark Sky" as well as a hyperlink leading to their website
import math
from datetime import datetime as dt
from darksky import forecast
from gitignore.API_keys import DARKSKYKEY2 as DARKSYKEY

# time required
# Either be a UNIX time (that is, seconds since midnight GMT on 1 Jan 1970) or a string formatted as follows:
# [YYYY]-[MM]-[DD]T[HH]:[MM]:[SS]

#the dict that is returned from the forecast call has the following format

# {
#     "latitude": 42.3601,
#     "longitude": -71.0589,
#     "timezone": "America/New_York",
#     "hourly": {
#         "summary": "Snow (6–9 in.) and windy starting in the afternoon.",
#         "icon": "snow",
#         "data": [
#             {
#                 "time": 255589200,
#                 "summary": "Mostly Cloudy",
#                 "icon": "partly-cloudy-night",
#                 "precipIntensity": 0,
#                 "precipProbability": 0,
#                 "temperature": 22.8,
#                 "apparentTemperature": 16.46,
#                 "dewPoint": 15.51,
#                 "humidity": 0.73,
#                 "pressure": 1026.78,
#                 "windSpeed": 4.83,
#                 "windBearing": 354,
#                 "cloudCover": 0.78,
#                 "uvIndex": 0,
#                 "visibility": 9.62
#             },
#             ...
#         ]
#     },
#     "daily": {
#         "data": [
#             {
#                 "time": 255589200,
#                 "summary": "Snow (9–14 in.) and windy starting in the afternoon.",
#                 "icon": "snow",
#                 "sunriseTime": 255613996,
#                 "sunsetTime": 255650764,
#                 "moonPhase": 0.97,
#                 "precipIntensity": 0.0354,
#                 "precipIntensityMax": 0.1731,
#                 "precipIntensityMaxTime": 255657600,
#                 "precipProbability": 1,
#                 "precipAccumulation": 7.337,
#                 "precipType": "snow",
#                 "temperatureHigh": 31.84,
#                 "temperatureHighTime": 255632400,
#                 "temperatureLow": 28.63,
#                 "temperatureLowTime": 255697200,
#                 "apparentTemperatureHigh": 20.47,
#                 "apparentTemperatureHighTime": 255625200,
#                 "apparentTemperatureLow": 13.03,
#                 "apparentTemperatureLowTime": 255697200,
#                 "dewPoint": 24.72,
#                 "humidity": 0.86,
#                 "pressure": 1016.41,
#                 "windSpeed": 22.93,
#                 "windBearing": 56,
#                 "cloudCover": 0.95,
#                 "uvIndex": 1,
#                 "uvIndexTime": 255621600,
#                 "visibility": 4.83,
#                 "temperatureMin": 22.72,
#                 "temperatureMinTime": 255596400,
#                 "temperatureMax": 32.04,
#                 "temperatureMaxTime": 255672000,
#                 "apparentTemperatureMin": 11.13,
#                 "apparentTemperatureMinTime": 255650400,
#                 "apparentTemperatureMax": 20.47,
#                 "apparentTemperatureMaxTime": 255625200
#             }
#         ]
#     },
#     "offset": -5
# }


def getPrecipitationChance(latitude, longitude, date_to_calculate):
    APIKEY=DARKSYKEY
    epoch_time = math.floor(date_to_calculate.timestamp())
    daily_forecast = forecast(APIKEY, latitude, longitude, epoch_time)
    precipitation_chance = daily_forecast.daily.data[0]['precipProbability']
    return precipitation_chance

def getWeatherData(latitude, longitude, date_to_calculate):
    APIKEY=DARKSYKEY
    epoch_time = math.floor(date_to_calculate.timestamp())
    daily_forecast = forecast(APIKEY, latitude, longitude, epoch_time)
    return daily_forecast