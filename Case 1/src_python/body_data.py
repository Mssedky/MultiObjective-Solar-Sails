import numpy as np
from .bodies_calc import *

def get_body_data(NEOname, dT, T):
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
    
    # Earth's Orbital Elements
    print("Calculating Earth's trajectory..")
    earthPos = np.empty((0, 3))
    earthVel = np.empty((0, 3))
    Earth = orbitingObject(1.496430050492096e11/AU, 0.01644732672533337, np.radians(0.002948905108822335), np.radians(250.3338397589344), np.radians(211.4594045093653),  np.radians(188.288909341482), 5.97219e24, 'Earth', earthPos, earthVel)
    Earth = calcCelestialTraj(Earth, dT, T)

    # NEO 2016 VO1 Orbital Elements
    print("Calculating NEO trajectory..")
    if NEOname == 'NEO':
        NEOPos = np.empty((0, 3))
        NEOVel = np.empty((0, 3))
        NEO = orbitingObject(146992149*1000/AU, 0.42869653, np.radians(1.61582328), np.radians(131.95199360416), np.radians(129.74129871507), np.radians(5.517059440296369e1), 0, 'NEO', NEOPos, NEOVel) 
        NEO = calcCelestialTraj(NEO, dT, T)
        targetObject = NEO

    # # 16 Psyche
    if NEOname == '16 Psyche':
        pschePos = np.empty((0, 3))
        psycheVel = np.empty((0, 3))
        Psyche = orbitingObject(4.372304796380328e11 / AU, 0.134153724002426, np.radians(3.092472666018101), np.radians(229.5568276039235), np.radians(150.019510189129), np.radians(302.7603308460203), 1.53, '16 Psyche', pschePos, psycheVel)
        Psyche = calcCelestialTraj(Psyche, dT, T)
        targetObject = Psyche

    # # Vesta 
    if NEOname == 'Vesta':
        vestaPos = np.empty((0, 3))
        vestaVel = np.empty((0, 3))
        Vesta = orbitingObject(3.531823140411854E+11 / AU, 8.996689130349783E-02, np.radians(7.14181494423702), np.radians(1.516915946723933E+02), np.radians(1.037050935401676E+02), np.radians(2.516482952688035E+02), 17.28824, 'Vesta', vestaPos, vestaVel)
        Vesta = calcCelestialTraj(Vesta, dT, T)
        targetObject = Vesta

    # Eunomia = orbitingObject(A, EC, IN, w, OM, MA, mass, name)
    if NEOname == 'Eunomia':
        eunomiaPos = np.empty((0, 3))
        eunomiaVel = np.empty((0, 3))
        Eunomia = orbitingObject(3.954262791032822E+11 / AU, 1.873532866281441E-01, np.radians(1.175476877128721E+01), np.radians(9.877585450769357E+01), np.radians(2.928995130001719E+02), np.radians(3.594995210917758E+02), 17.28824, 'Eunomia', eunomiaPos, eunomiaVel)
        Eunomia = calcCelestialTraj(Eunomia, dT, T)
        targetObject = Eunomia

    # Ceres = orbitingObject(A, EC, IN, w, OM, MA, mass, name)
    if NEOname == 'Ceres':
        ceresPos = np.empty((0, 3))
        ceresVel = np.empty((0, 3))
        Ceres = orbitingObject(4.139129887889629E+11 / AU, 7.910717043218352E-02, np.radians(1.058782499528511E+01), np.radians(7.331580895116618E+01), np.radians(8.025383722598423E+01), np.radians(1.250473243162762E+02), 17.28824, 'Ceres', ceresPos, ceresVel)
        Ceres = calcCelestialTraj(Ceres, dT, T)
        targetObject = Ceres

    # Bennu = orbitingObject(A, EC, IN, w, OM, MA, mass, name)
    if NEOname == 'Bennu':
        print("Calculating Bennu trajectory...")
        bennuPos = np.empty((0, 3))
        bennuVel = np.empty((0, 3))
        Bennu = orbitingObject(1.684403508572353E+11 / AU, 2.037483028559170E-01, np.radians(6.032932274441114E+00), np.radians(6.637388139157433E+01), np.radians(1.981305199928344E+00), np.radians(2.175036361198920E+02), 7.329E10, 'Bennu', bennuPos, bennuVel) #mass 7.329E10 from: https://www.nature.com/articles/s41550-019-0721-3
        Bennu = calcCelestialTraj(Bennu, dT, T)
        targetObject = Bennu

    bodies = {Earth, targetObject}

    return bodies