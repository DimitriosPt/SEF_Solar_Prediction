from datetime import datetime as dt
import math
from math import radians as rad
from math import cos as cos, sin as sin
import numpy
from numpy import dot as dot, cross as cross
import pytz
PI = 3.14159
TILT_VARIATION = PI * 2 / 365  # W0
PERIOD_OF_DAYS = PI * 2 / 24  # W1
EARTH_TILT = math.radians(22.5)  # E0
PANEL_TILT_VERTICAL = 25
PANEL_TILT_HORIZONTAL = 0

def calculate_atmospheric_attenuation(latitude, date_to_calculate):
    summer_solstice = dt(month=6, day=21, year=date_to_calculate.year, tzinfo=pytz.timezone('US/Pacific'))

    date_to_calculate = date_to_calculate.replace(tzinfo=pytz.timezone('US/Pacific'))
    # if you have a date like march 31st 2019 it would give a negative date
    # this if statement will make it so 3/31/19 compares to solstice of 6/21/18
    if (date_to_calculate - summer_solstice).days < 0:
        summer_solstice = dt(month=6, day=21, year=(summer_solstice.year - 1))

    days_since_summer_sols = (date_to_calculate - summer_solstice).days
    earth_offset = -EARTH_TILT * cos(TILT_VARIATION * days_since_summer_sols)
    rotation_axis = numpy.array([-sin(earth_offset), 0, cos(earth_offset)])
    position_at_midday = numpy.array([cos(earth_offset + latitude), 0, sin(earth_offset + latitude)])
    midnight = dt(month=date_to_calculate.month, day=date_to_calculate.day,
                  year=date_to_calculate.year, hour=0, minute=0)

    minutes_from_midnight = ((date_to_calculate - pytz.timezone('US/Pacific').localize(midnight)).seconds / 60)
    t = minutes_from_midnight / 60 - 12
    gamma = PERIOD_OF_DAYS * t
    position_of_observer = cos(gamma) * position_at_midday + sin(gamma) * cross(rotation_axis, position_at_midday) \
                           + (dot(rotation_axis, position_at_midday) * ((1 - cos(gamma)) * rotation_axis))

    zenith = math.acos(dot(position_of_observer, numpy.array([1, 0, 0])))

    if zenith < (PI / 2):
        return numpy.exp((cos(zenith) - 1) / cos(zenith))

    else:
        return 0
