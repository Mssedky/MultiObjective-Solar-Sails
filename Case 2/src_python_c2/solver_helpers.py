import numpy as np
from scipy.optimize import root
from scipy.interpolate import lagrange
import pandas as pd

def initialPos(xE,yE,rInit):
    # Solves two equations in two unknowns to determine the x and y positions of the light sail on Earth's orbit  
    result = root(lambda variables: [(variables[0]-xE) * xE + (variables[1]-yE) * yE,
                                     (variables[0] - xE)**2 + (variables[1] - yE)**2 - rInit**2], 
                  [xE + rInit, yE], method='hybr')  # Initial guess for xS and yS 

    xS, yS = result.x

    return xS,yS

def calc_time_segments(time_var):
    return np.cumsum(np.insert(time_var,0,0))

def parse_angles(time_segments, cone_angle_var, clock_angle_var, t_days):
    ind = np.searchsorted(time_segments,t_days,side="right")
    if ind < len(time_segments)-1:
        return cone_angle_var[ind], clock_angle_var[ind]
    else:
        return 0,0

# Function to create angle polynomials for each segment
def create_angle_functions(time_segments, clock_angles, cone_angles, degree):

    segmentsClocks=[]
    segmentsCones=[]

    i = 0
    n=len(clock_angles)

    while i < n - 1:
        
        # Determine the number of points available to fit a polynomial
        points_remaining = n - i
        degree_to_fit = min(degree, points_remaining - 1)
        
        # Cap the degree to avoid instability
        if degree_to_fit > 4:
            degree_to_fit = 4  # Restrict to degree 4 to avoid overfitting

        # Number of points needed to fit a polynomial of degree `degree_to_fit`
        points_needed = degree_to_fit + 1
        
        # Select the points for this segment, including the overlap of the last point
        clocks = clock_angles[i:i + points_needed]
        cones = cone_angles[i:i + points_needed]
        times = time_segments[i:i+points_needed]
        
        # Fit an exact polynomial to clock angles using Lagrange interpolation
        polyClocks = lagrange(times, clocks)
        segmentsClocks.append((times, clocks, polyClocks))

        # Fit an exact polynomial to cone angles using Lagrange interpolation
        polyCones = lagrange(times, cones)
        segmentsCones.append((times, cones, polyCones))
        
        i += degree_to_fit

    return segmentsClocks, segmentsCones 

def find_value_at_time(angles,time_segments, t_days):
    ind = np.searchsorted(time_segments,t_days,side="right")
    if ind < len(time_segments)-1:
         for times, values, poly in angles:
            if times.min() <= t_days <= times.max():
                return poly(t_days)
    else:
        return 0

def find_body(bodies, name):
    for body in bodies:
        if body.name == name:
            return body
    return None