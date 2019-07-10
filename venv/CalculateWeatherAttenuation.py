# Dimitrios Papageorgacopoulos
# Intern at Stone Edge Farms Microgrid Project
# June 21 2019

# this script uses the Dark Sky api
# documentation for the wrapper be found here https://pypi.org/project/darkskylib/
# documentation for the dark sky api is found here : https://darksky.net/dev/docs
from darksky import forecast
import math
from datetime import datetime as dt

# time required
# Either be a UNIX time (that is, seconds since midnight GMT on 1 Jan 1970) or a string formatted as follows:
# [YYYY]-[MM]-[DD]T[HH]:[MM]:[SS]
def calculateWeatherAttenuation(latitude, longitude, date_to_calculate):
    DARKSKYKEY = '79b72a42c4e643927e13401556084447'
    #date_to_calculate = dt(date_to_calculate)
    epoch_time = math.floor(date_to_calculate.timestamp())
    daily_forecast = forecast(DARKSKYKEY, latitude, longitude, epoch_time)
    precipitation_chance = daily_forecast.daily.data[0]['precipProbability']
    return precipitation_chance
