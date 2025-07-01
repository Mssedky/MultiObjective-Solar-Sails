import numpy as np
from .bodies_calc import *

def get_body_data(dT, T):
    AU = 1.496e11 # Astronomical Unit (m)

    '''
        a = semi_major_axis # Semi-major axis of Earth's orbit (AU)
        e = eccentricity # Eccentricity of Earth's orbit
        i = inclination # Inclination (radians)
        omega = periapsis_arg # Argument of periapsis (radians)
        Omega = longitude_asc_node # Longitude of ascending node (radians)
        nuMean=mean_anomaly # Approximate mean anomaly on February 24, 2023 (radians)
        (a, e, i, omega, Omega, nuMean, mass, name, positions, velocities)
        '''
    
    # ALL CELESTIAL BODY DATA BELOW IS FOR JULY 12, 2024
    print("Calculating Planet Trajectories...")

    # Earth's Orbital Elements
    earthPos = np.empty((0, 3))
    earthVel = np.empty((0, 3))
    Earth = orbitingObject(1.496430050492096e11/AU, 0.01644732672533337, np.radians(0.002948905108822335), np.radians(250.3338397589344), np.radians(211.4594045093653),  np.radians(188.288909341482), 5.97219e24, 'Earth', earthPos, earthVel)
    Earth = calcCelestialTraj(Earth, dT, T)
    
    # Mercury Orbital Elements
    mercuryPos = np.empty((0, 3))
    mercuryVel = np.empty((0, 3))
    Mercury = orbitingObject(5.790890474396899e10/AU, 0.2056491892618194, np.radians(7.003540550842122), np.radians(29.1947223267306), np.radians(48.30037576148549), np.radians(115.9712246827886), 3.302e23, 'Mercury', mercuryPos, mercuryVel)
    Mercury = calcCelestialTraj(Mercury, dT, T)

    # Venus Orbital Elements
    venusPos = np.empty((0, 3))
    venusVel = np.empty((0, 3))
    Venus = orbitingObject(1.082082396106793e11/AU, 0.006731004188873255, np.radians(3.394392365364129),  np.radians(55.19971076495922), np.radians(76.61186125621185), np.radians(2.854406665289592), 48.685e23, 'Venus', venusPos, venusVel)
    Venus = calcCelestialTraj(Venus, dT, T)

    # Mars Orbital Elements
    marsPos = np.empty((0, 3))
    marsVel = np.empty((0, 3))
    Mars = orbitingObject(2.279365490986187e11/AU, 0.09329531518708468, np.radians(1.847838824432797), np.radians(286.6931548864934), np.radians(49.4895026740929), np.radians(33.81804161260958), 6.4171e23, 'Mars', marsPos, marsVel)
    Mars = calcCelestialTraj(Mars, dT, T)

    # Jupiter Orbital Elements
    jupiterPos = np.empty((0, 3))
    jupiterVel = np.empty((0, 3))
    Jupiter = orbitingObject(7.783250081012725e11/AU, 0.04828195024989049, np.radians(1.303342363753197), np.radians(273.6372929361556), np.radians(100.5258994533003), np.radians(44.57901055693271), 1.89818722e27, 'Jupiter', jupiterPos, jupiterVel)
    Jupiter = calcCelestialTraj(Jupiter, dT, T)

    # Saturn Orbital Elements
    saturnPos = np.empty((0, 3))
    saturnVel = np.empty((0, 3))
    Saturn = orbitingObject(1.430659999993930e12/AU, 0.05487028713273947, np.radians(2.487163740030422),  np.radians(336.3262942809200), np.radians(113.6081294721804), np.radians(259.9865288607137), 5.6834e26, 'Saturn', saturnPos, saturnVel)
    Saturn = calcCelestialTraj(Saturn, dT, T)

    # Uranus Orbital Elements
    uranusPos = np.empty((0, 3))
    uranusVel = np.empty((0, 3))
    Uranus = orbitingObject(2.887674772169614e12/AU, 0.04505591744341015, np.radians(0.7721952479077103), np.radians(90.56594589691922), np.radians(74.03018776280420), np.radians(253.7189038328262), 86.813e24, 'Uranus', uranusPos, uranusVel)
    Uranus = calcCelestialTraj(Uranus, dT, T)

    # Neptune Orbital Elements
    neptunePos = np.empty((0, 3))
    neptuneVel = np.empty((0, 3))
    Neptune = orbitingObject(4.520641437530186e12/AU, 0.01338220205649052, np.radians(1.772753210867184), np.radians(264.0854134905526), np.radians(131.8710844903620), np.radians(322.7338947558696), 1.02409e26, 'Neptune', neptunePos, neptuneVel)
    Neptune = calcCelestialTraj(Neptune, dT, T)
    
    print("Done Calculating.")

    bodies = {Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune}

    return bodies


def get_full_body_trajectories(dT):
    AU = 1.496e11 # Astronomical Unit (m)

    # Earth's Orbital Elements
    earthPos2 = np.empty((0, 3))
    earthVel2 = np.empty((0, 3))
    Earth2 = orbitingObject(1.496430050492096e11/AU, 0.01644732672533337, np.radians(0.002948905108822335), np.radians(250.3338397589344), np.radians(211.4594045093653),  np.radians(188.288909341482), 5.97219e24, 'Earth', earthPos2, earthVel2)
    Earth2 = calcCelestialTraj(Earth2, dT, 365*5, tTraj = 0)
    
    # Mercury Orbital Elements
    mercuryPos2 = np.empty((0, 3))
    mercuryVel2 = np.empty((0, 3))
    Mercury2 = orbitingObject(5.790890474396899e10/AU, 0.2056491892618194, np.radians(7.003540550842122), np.radians(29.1947223267306), np.radians(48.30037576148549), np.radians(115.9712246827886), 3.302e23, 'Mercury', mercuryPos2, mercuryVel2)
    Mercury2 = calcCelestialTraj(Mercury2, dT, 365*5, tTraj = 0)

    # Venus Orbital Elements
    venusPos2 = np.empty((0, 3))
    venusVel2 = np.empty((0, 3))
    Venus2 = orbitingObject(1.082082396106793e11/AU, 0.006731004188873255, np.radians(3.394392365364129),  np.radians(55.19971076495922), np.radians(76.61186125621185), np.radians(2.854406665289592), 48.685e23, 'Venus', venusPos2, venusVel2)
    Venus2 = calcCelestialTraj(Venus2, dT, 365*5, tTraj = 0)

    # Mars Orbital Elements
    marsPos2 = np.empty((0, 3))
    marsVel2 = np.empty((0, 3))
    Mars2 = orbitingObject(2.279365490986187e11/AU, 0.09329531518708468, np.radians(1.847838824432797), np.radians(286.6931548864934), np.radians(49.4895026740929), np.radians(33.81804161260958), 6.4171e23, 'Mars', marsPos2, marsVel2)
    Mars2 = calcCelestialTraj(Mars2, dT, 365*5, tTraj = 0)

    # Jupiter Orbital Elements
    jupiterPos2 = np.empty((0, 3))
    jupiterVel2 = np.empty((0, 3))
    Jupiter2 = orbitingObject(7.783250081012725e11/AU, 0.04828195024989049, np.radians(1.303342363753197), np.radians(273.6372929361556), np.radians(100.5258994533003), np.radians(44.57901055693271), 1.89818722e27, 'Jupiter', jupiterPos2, jupiterVel2)
    Jupiter2 = calcCelestialTraj(Jupiter2, dT, 365*200, tTraj = 0)

    # Saturn Orbital Elements
    saturnPos2 = np.empty((0, 3))
    saturnVel2 = np.empty((0, 3))
    Saturn2 = orbitingObject(1.430659999993930e12/AU, 0.05487028713273947, np.radians(2.487163740030422),  np.radians(336.3262942809200), np.radians(113.6081294721804), np.radians(259.9865288607137), 5.6834e26, 'Saturn', saturnPos2, saturnVel2)
    Saturn2 = calcCelestialTraj(Saturn2, dT, 365*200, tTraj = 0)

    # Uranus Orbital Elements
    uranusPos2 = np.empty((0, 3))
    uranusVel2 = np.empty((0, 3))
    Uranus2 = orbitingObject(2.887674772169614e12/AU, 0.04505591744341015, np.radians(0.7721952479077103), np.radians(90.56594589691922), np.radians(74.03018776280420), np.radians(253.7189038328262), 86.813e24, 'Uranus', uranusPos2, uranusVel2)
    Uranus2 = calcCelestialTraj(Uranus2, dT, 365*200, tTraj = 0)

    # Neptune Orbital Elements
    neptunePos2 = np.empty((0, 3))
    neptuneVel2 = np.empty((0, 3))
    Neptune2 = orbitingObject(4.520641437530186e12/AU, 0.01338220205649052, np.radians(1.772753210867184), np.radians(264.0854134905526), np.radians(131.8710844903620), np.radians(322.7338947558696), 1.02409e26, 'Neptune', neptunePos2, neptuneVel2)
    Neptune2 = calcCelestialTraj(Neptune2, dT, 365*200, tTraj = 0)

    return Mercury2, Venus2, Earth2, Mars2, Jupiter2, Saturn2, Uranus2, Neptune2