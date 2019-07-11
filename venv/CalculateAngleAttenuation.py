from datetime import datetime as dt
import math
from math import cos as cos
from math import sin as sin
import numpy
from numpy import cross as cross
from numpy import dot as dot
from numpy import linalg
import pytz

def calculate_angle_between_panel_sun(latitude, date_to_calculate):
    PI = 3.14159
    TILT_VARIATION = PI * 2 / 365  # W0
    PERIOD_OF_DAYS = PI * 2 / 24  # W1
    EARTH_TILT = math.radians(22.5)  # E0
    PANEL_TILT_VERTICAL = 25
    PANEL_TILT_HORIZONTAL = 0

    alpha = PANEL_TILT_VERTICAL * PI / 180
    beta = PANEL_TILT_HORIZONTAL * PI / 180

    # convert date given to datetime object
    # date_to_calculate = dt.date(date_to_calculate)
    summer_solstice = dt(month=6, day=21, year=date_to_calculate.year)
    summer_solstice = summer_solstice.replace(tzinfo=pytz.timezone('US/Pacific'))
    date_to_calculate = date_to_calculate.replace(tzinfo=pytz.timezone('US/Pacific'))
    # if you have a date like march 31st 2019 it would give a negative date
    # this if statement will make it so 3/31/19 compares to solstice of 6/21/18
    if (date_to_calculate - summer_solstice).days < 0:
        summer_solstice = dt(month=6, day=21, year=(summer_solstice.year - 1))

    days_since_summer_sols = (date_to_calculate - summer_solstice).days

    #revisit math to see if this should be radians
    earth_offset = math.radians(-EARTH_TILT * cos(TILT_VARIATION * days_since_summer_sols))
    position_at_midday = numpy.array([cos(earth_offset + latitude), 0, sin(earth_offset + latitude)])
    rotation_axis = numpy.array([-sin(earth_offset), 0, cos(earth_offset)])

    # The original matlab code uses a variable t which is effectively how many hundredths of an hour have
    # passed since midnight as opposed to a conventional time interval
    midnight = dt(month=date_to_calculate.month, day=date_to_calculate.day,
                  year=date_to_calculate.year, hour=0, minute=0)
    minutes_from_midnight = ((date_to_calculate - pytz.timezone('US/Pacific').localize(midnight)).seconds / 60)

    t = minutes_from_midnight / 60 - 12

    gamma = PERIOD_OF_DAYS * t
    # position at midday is a
    position_of_observer = (cos(gamma) * position_at_midday) + (sin(gamma) * cross(rotation_axis, position_at_midday)) \
                           + (dot(rotation_axis, position_at_midday) * (1 - cos(gamma)) * rotation_axis)

    rotation_about_y = numpy.array([0, 1, 0])
    c = cos(earth_offset) * position_of_observer + sin(earth_offset) * cross(rotation_about_y, position_of_observer) + \
        dot(rotation_about_y, position_of_observer) * (1 - cos(earth_offset)) * rotation_about_y
    theta = math.atan2(dot(c, numpy.array([0, 1, 0])), dot(c, numpy.array([1, 0, 0])))
    phi = math.acos(dot(c, numpy.array([0, 0, 1])))

    south_pole_vector = numpy.array([cos(theta) * cos(phi), sin(theta) * cos(phi), -sin(phi)])
    r_three = numpy.array([0, -1, 0])

    S = cos(earth_offset) * south_pole_vector + sin(earth_offset) * cross(r_three, south_pole_vector) + \
        dot(r_three, south_pole_vector) * (1 - cos(earth_offset)) * r_three

    zenith = math.acos(dot(position_of_observer, numpy.array([1, 0, 0])))

    projection_of_sun_horizontal = (numpy.array([1, 0, 0]) - dot(position_of_observer, numpy.array([1, 0, 0]))
                                    * position_of_observer)

    # why not just do greater than 0?
    if linalg.norm(projection_of_sun_horizontal) < 1 * 10 ^ -6:
        delta = 0

    else:
        delta = numpy.real(math.acos(dot(S, projection_of_sun_horizontal))
                           / linalg.norm(projection_of_sun_horizontal))

        if dot(cross(S, projection_of_sun_horizontal), position_of_observer) < 0:
            delta = delta * -1

    direction_of_sun = numpy.array([cos(delta) * sin(zenith), sin(delta) * sin(zenith), cos(zenith)])
    panel_orientation = numpy.array([cos(beta) * sin(alpha), sin(beta) * sin(alpha), cos(alpha)])

    lower_gamma = math.acos(dot(direction_of_sun, panel_orientation))

    if dot(cross(direction_of_sun, panel_orientation), position_of_observer) < 0:
        lower_gamma = lower_gamma * -1

    if zenith < (PI / 2):
        angle_between_panel_sun = max(cos(lower_gamma), 0)
    else:
        angle_between_panel_sun = 0

    return angle_between_panel_sun
