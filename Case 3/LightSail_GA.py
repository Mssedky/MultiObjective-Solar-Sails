import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fsolve
from scipy.optimize import root
import time
import argparse
from autograd import grad
import autograd.numpy as np  
import itertools

from LightSailSolver import *
from BodiesCalc import *


def generate_smooth_angles(num_angles, lb, ub, max_variation):
    angles = np.zeros(num_angles)
    angles[0] = np.random.uniform(lb[0], ub[0])
    for i in range(1, num_angles):
        min_angle = max(lb[i], angles[i-1] - max_variation)
        max_angle = min(ub[i], angles[i-1] + max_variation)
        angles[i] = np.random.uniform(min_angle, max_angle)
    return angles

def GeneticAlgorithm(S,P,K,TOL,G,dv,lb,ub,func, desHoverTime, constant_angles, T, w, TOLNEO,TOLEarth, max_variation, numSeg, dT, bodies, printStatements = 1, NEOname = "Vesta"):
    cost=np.ones((S,1))*1000
    prev_cost = None
    Pi=np.empty((0,S))
    meanParents=[]
    Orig=np.zeros((G,S))
    Children=np.zeros((K,dv))
    Parents=np.zeros((P,dv))
    Lambda=np.zeros((S,dv))
    Gen=1
    start = 0

    # #Generate starting population
    # pop_new=np.random.uniform(lb, ub, (S, dv))
    # pop_new[:, 0] = np.random.randint(lb[0], ub[0] + 1, S)  # Ensure degree is an integer
    
    # Generate starting population
    if max_variation == 0:
        pop_new=np.random.uniform(lb, ub, (S, dv))
    else:
        pop_new = np.zeros((S, dv))
        for i in range(S):
            pop_new[i, :3+numSeg] = np.random.uniform(lb[:3+numSeg], ub[:3+numSeg])
            pop_new[i, 3+numSeg:3+2*numSeg] = generate_smooth_angles(numSeg, lb[3+numSeg:], ub[3+numSeg:], max_variation)
            pop_new[i, 3+2*numSeg:3+3*numSeg] = generate_smooth_angles(numSeg, lb[3+2*numSeg:], ub[3+2*numSeg:], max_variation)

    pop_new[:, 0] = np.random.randint(lb[0], ub[0] + 1, S)  # Ensure degree is an integer
    

    while np.abs(np.min(cost))>TOL and Gen<G:
        if printStatements == 1:
            print("**********************************")
            print("Generation number : ", Gen)
        pop=pop_new

        #Evaluate population fitness
        for i in range(start, S):
            if printStatements == 1:
                print(f"String : {i+1}, Gen : {Gen} ")
            cost[i]=func(pop[i,:], desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname = NEOname)

        #Sort population fitnesses
        Index=np.argsort(cost[:,0])
        pop=pop[Index,:]
        cost=cost[Index,:]

        if printStatements == 1:
            print(f"Best cost for generation {Gen} : {cost[0]}")
        # print(f"Best cost occurs at initial launch time= {pop[0,0]} days and velocity={pop[0,1]*1.496e11} m/s")

        #Select parents
        Parents=pop[0:P,:]
        meanParents.append(np.mean(cost[0:P]))


        #Generate K offspring
        for i in range(0,K,2):
            #Breeding parents
            alpha=np.random.uniform(0,1)
            beta=np.random.uniform(0,1)
            Children[i, :] = Parents[i, :] * alpha + Parents[i + 1, :] * (1 - alpha)
            Children[i + 1, :] = Parents[i, :] * beta + Parents[i + 1, :] * (1 - beta)

            # Ensure degree is an integer
            Children[i, 0] = int(np.round(Children[i, 0]))
            Children[i + 1, 0] = int(np.round(Children[i + 1, 0]))

            # Clip angles to ensure they are within bounds
            Children[i, :] = np.clip(Children[i, :], lb, ub)
            Children[i + 1, :] = np.clip(Children[i + 1, :], lb, ub)

        
        #Overwrite population with P parents, K children, and S-P-K random values
        random_values = np.random.uniform(lb, ub, (S - P - K, dv))
        random_values[:, 0] = np.random.randint(lb[0], ub[0] + 1, S - P - K)  # Ensure degree is an integer
        pop_new = np.vstack((Parents, Children, random_values))

        #Store costs and indices for each generation
        Pi= np.vstack((Pi, cost.T))
        Orig[Gen,:]=Index
        #Increment generation counter
        Gen=Gen+1
        start = P

    #Store best population 
    Lambda=pop    
    meanPi=np.mean(Pi,axis=1)
    minPi=np.min(Pi,axis=1)
    return Lambda, Pi, Orig, meanPi, minPi,meanParents,cost

def GeneticAlgorithmSolarEscape(S, P, K, TOL, G, dv, lb, ub, func, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, max_variation, numSeg, dT, bodies, numSails, printStatements=1):
    # Adjust bounds for all sails
    dv_total = dv * numSails  # Total dimensionality
    lb = np.tile(lb, numSails)
    ub = np.tile(ub, numSails)

    cost = np.ones((S, 1)) * 1000
    prev_cost = None
    Pi = np.empty((0, S))
    meanParents = []
    Orig = np.zeros((G, S))
    Children = np.zeros((K, dv_total))
    Parents = np.zeros((P, dv_total))
    Lambda = np.zeros((S, dv_total))
    Gen = 1
    start = 0

   
    # Generate starting population
    if max_variation == 0:
        pop_new = np.random.uniform(lb, ub, (S, dv_total))
    else:
        pop_new = np.zeros((S, dv_total))
        for i in range(S):
            for sail in range(numSails):
                start_idx = sail * dv
                pop_new[i, start_idx:start_idx + 3 + numSeg] = np.random.uniform(lb[:3 + numSeg], ub[:3 + numSeg])
                pop_new[i, start_idx + 3 + numSeg:start_idx + 3 + 2 * numSeg] = generate_smooth_angles(numSeg, lb[3 + numSeg:], ub[3 + numSeg:], max_variation)
                pop_new[i, start_idx + 3 + 2 * numSeg:start_idx + 3 + 3 * numSeg] = generate_smooth_angles(numSeg, lb[3 + 2 * numSeg:], ub[3 + 2 * numSeg:], max_variation)

    pop_new[:, 0::dv] = np.random.randint(lb[0], ub[0] + 1, (S, numSails))  # Ensure degree is an integer
        

    while np.abs(np.min(cost)) > TOL and Gen < G:
        if printStatements == 1:
            print("**********************************")
            print("Generation number : ", Gen)
        pop = pop_new

        # Evaluate population fitness
        for i in range(start, S):
            if printStatements == 1:
                print(f"String : {i+1}, Gen : {Gen}")
            cost[i] = func(pop[i, :], desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, numSails)

        # Sort population fitnesses
        Index = np.argsort(cost[:, 0])
        pop = pop[Index, :]
        cost = cost[Index, :]

        if printStatements == 1:
            print(f"Best cost for generation {Gen} : {cost[0]}")

        # Select parents
        Parents = pop[0:P, :]
        meanParents.append(np.mean(cost[0:P]))

        # Generate K offspring
        for i in range(0, K, 2):
            # Breeding parents
            alpha = np.random.uniform(0, 1)
            beta = np.random.uniform(0, 1)
            Children[i, :] = Parents[i, :] * alpha + Parents[i + 1, :] * (1 - alpha)
            Children[i + 1, :] = Parents[i, :] * beta + Parents[i + 1, :] * (1 - beta)

            # Ensure degree is an integer
            Children[i, 0::dv] = np.round(Children[i, 0::dv])
            Children[i + 1, 0::dv] = np.round(Children[i + 1, 0::dv])

            # Clip angles to ensure they are within bounds
            Children[i, :] = np.clip(Children[i, :], lb, ub)
            Children[i + 1, :] = np.clip(Children[i + 1, :], lb, ub)

        # Overwrite population with P parents, K children, and S-P-K random values
        random_values = np.random.uniform(lb, ub, (S - P - K, dv_total))
        random_values[:, 0::dv] = np.random.randint(lb[0], ub[0] + 1, (S - P - K, numSails))  # Ensure degree is an integer
        pop_new = np.vstack((Parents, Children, random_values))

        # Store costs and indices for each generation
        Pi = np.vstack((Pi, cost.T))
        Orig[Gen, :] = Index
        # Increment generation counter
        Gen += 1
        start = P

    # Store best population
    Lambda = pop
    meanPi = np.mean(Pi, axis=1)
    minPi = np.min(Pi, axis=1)
    return Lambda, Pi, Orig, meanPi, minPi, meanParents, cost

def lightSailCost(var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname = "Vesta"):

    solver = lightSailSolver(var, bodies)
    solver.runSim(desHoverTime, constant_angles, T, TOLNEO, TOLEarth, MakeMovie, NumSeg, dT, NEOname = NEOname)
    cost = solver.calcCost(w,TOLNEO,TOLEarth)

    return cost

def solarExitCost(var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, numSails):
    finalPositions = []
    sunCosts = []
    distCosts = []
    posCosts = []

    for i in range(numSails):
        solver = lightSailSolver(np.concatenate(([var[i*dv], var[1]], var[i*dv + 2:(i+1)*dv])), bodies)
        # solver = lightSailSolver(var[i*dv, 1, i*dv + 2 :(i+1)*dv], bodies)
        solver.runSim(desHoverTime, constant_angles, T, TOLNEO, TOLEarth, MakeMovie, NumSeg, dT)
        cost = solver.calcCostSolarExit()

        sunCosts.append(cost[0])
        distCosts.append(cost[1])
        finalPositions.append(cost[2])
        
    finalPositions = np.array(finalPositions)
    
    # Calculate pairwise distances and find the smallest distance for each sail
    for i in range(numSails):
        distances = np.linalg.norm(finalPositions[i] - finalPositions, axis=1)
        # Exclude the distance to itself by setting it to a large number
        distances[i] = np.inf
        min_distance = np.min(distances)
        posCosts.append(min_distance)

    finalCost = w[0] * np.max(sunCosts) + w[1] * np.min(distCosts) - w[2] * np.min(posCosts)

    print("Cost contribution 1 (Closest Distance to Sun):", w[0] * np.sum(sunCosts))
    print("Cost contribution 2 (Distance away from Sun):", w[1] * np.min(distCosts))
    print("Cost contribution 3 (Distances from each other):", - w[2] * np.min(posCosts))
    print("Total cost: ", finalCost)

    return finalCost

def bruteForceOptimization(var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, max_iter=20, tolerance=150, m=0.2, printStatements = 1, NEOname = "Vesta"):
    cost = lightSailCost(var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname = NEOname)
    no_change = 0

    for j in range(max_iter):
        for i in range(1,len(var)):
            if no_change > tolerance:
                m = m * 0.5
                no_change = 0

            inc_var = np.copy(var)
            inc_var[i] = inc_var[i] + inc_var[i] * m
            dec_var = np.copy(var)
            dec_var[i] = dec_var[i] - dec_var[i] * m

            inc_var_cost = lightSailCost(inc_var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname = NEOname)
            dec_var_cost = lightSailCost(dec_var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname = NEOname)

            if cost > inc_var_cost:
                var = inc_var
                cost = inc_var_cost
                no_change = 0
            else:
                no_change += 1

            if cost > dec_var_cost:
                var = dec_var
                cost = dec_var_cost
                no_change = 0
            else:
                no_change += 1
            
            if printStatements == 1:
                print("var #:", i, "m", m, "new cost:", cost)

        if cost < 0:
            break

    return var, cost


if __name__ == "__main__":

    dT_default = 5 #default time step = 5 days
    randomseed_default = 0 #default randomseed is 0
    bruteForce_default = 1 #default: don't run bruteForce
    experiments_default = 0 # default: don't run long experiments
    solarSysExit_default = 1 #default: use solar sys exit version
    T_default = 365*15 #default: max simulation time is 10 years
    NEOname_default = 'Vesta' #default: target NEO object is Vesta

    parser = argparse.ArgumentParser(
                        prog='LightSail_GA.py',
                        description='Genetic algorithm for single trajectory optimization')
    parser.add_argument('-f', '--filename')       # filename for saving final optimized var -- if no filename given, no save
    parser.add_argument('-rs', '--randomseed')    # random seed value
    parser.add_argument('-dT', '--dT')            # time step size (days)
    parser.add_argument('-b', '--bruteforce')     # bruteforce optimization option -- run if 1
    parser.add_argument('--solarSysExit')         # run solar system exit version of code if =1
    parser.add_argument('--T')                    # maximum simulation time (days)
    parser.add_argument('--experiments')          # Set up experiment to determine best combination of segment lengths, max variations, constant angles, and number of segments (VERY LONG TIME TO RUN)
    parser.add_argument('--NEOname') # Name of target object for simulation. options: 'NEO', '16 Psyche', 'Vesta', 'Eunomia', 'Ceres'
    

    args = parser.parse_args()

    dT = args.dT if args.dT!=None else dT_default
    seed = args.randomseed if args.randomseed!=None else randomseed_default
    bruteForce = int(args.bruteforce) if args.bruteforce!=None else bruteForce_default
    solarSysExit = int(args.solarSysExit) if args.solarSysExit!=None else solarSysExit_default
    T = float(args.T) if args.T!=None else T_default
    experiments = int(args.experiments) if args.experiments!=None else experiments_default
    NEOname = args.NEOname if args.NEOname!=None else NEOname_default


    np.random.seed(int(seed))

    AU = 1.496e11 # Astronomical Unit (m)
    MakeMovie = 0
    # T = 365*15 # Maximum simulation time (days) 
    desHoverTime = 60 # Desired hover time (days) 

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
    print("Calculating Earth trajectory..")
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

    if solarSysExit == 1:
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

        bodies = {Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune}
    else:
        bodies = {Earth, targetObject}

    #Define genetic algorithm parameters
    S=30
    P=10
    K=10
    G=50
    TOL=1e-3
    
    if  experiments != 1: 
        constant_angles = 0 # constant vs smooth angles

        #Design Variables min and max
        NumSeg = 30
        max_variation = 0 # Set to 0 for regular / Set to any other value for "smooth" angle variation
        degree_min = 1
        degree_max = 5
        time_min = 10
        time_max = 365
        vel_min=30*1000/AU # (AU/s)
        vel_max=30*1000/AU # (AU/s)
        setInitDir_min = 0
        setInitDir_max = 1
        timeSeg_min=np.full(NumSeg,1)
        timeSeg_max=np.full(NumSeg,80)
        cone_angles_min=np.full(NumSeg,-1.2)
        cone_angles_max=np.full(NumSeg,1.2)
        clock_angles_min=np.full(NumSeg,-6.2831/2)
        clock_angles_max=np.full(NumSeg,6.2831/2)

        if solarSysExit != 1:

            #Number of design variables
            dv=3+NumSeg*3

            #Upper and lower bounds for each variable
            lb = np.concatenate([[degree_min, time_min, vel_min],timeSeg_min,cone_angles_min,clock_angles_min])
            ub = np.concatenate([[degree_max, time_max, vel_max],timeSeg_max,cone_angles_max,clock_angles_max])

            #Desired designs and weights
            w = [0] * 6
            w[0] = 1 # Total energy
            w[1] = 1 # Hover Time
            # w[1] = 10 # Hover Time
            w[2] = 10 # Hover Time 2
            w[3] = 0 # Return
            w[4] = 1 # Closest Distance to Sun
            w[5] = 10 # Approach velocity to NEO

            TOLEarth=0.1
            TOLNEO=100000*1000/AU

            # Record the start time
            start_time = time.time()
            
            #Call genetic algorithm to mininmize cost function
            # np.random.seed(11)
            Lambda,Pi,Orig,meanPi,minPi,meanParents,costs=GeneticAlgorithm(S,P,K,TOL,G,dv,lb,ub,lightSailCost, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, max_variation, NumSeg, dT, bodies, NEOname = NEOname)
            var = Lambda[0,:]

            print("***************************************************************")
            print("The design variables are:")
            np.set_printoptions(threshold=1000000000)
            print(np.array2string(var, separator=', '))
            print("***************************************************************")
            print("The corresponding costs are :")
            print(lightSailCost(var,desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname = NEOname))
            print("The minimum cost is: ", costs[0])
            print("***************************************************************")
        
        else:
            # Number of sails in the swarm
            numSails = 30

            #Number of design variables
            dv=3+NumSeg*3

            #Upper and lower bounds for each variable
            lb = np.concatenate([[degree_min, time_min, vel_min],timeSeg_min,cone_angles_min,clock_angles_min])
            ub = np.concatenate([[degree_max, time_max, vel_max],timeSeg_max,cone_angles_max,clock_angles_max])

            #Desired designs and weights
            w = [0] * 3
            w[0] = 1 # Closest distance to Sun penalty
            w[1] = 10 # Distance from Sun
            w[2] = 0.1 # Distances between sails
     
            TOLEarth=0.1
            TOLNEO=100000*1000/AU
            # TOLNEO=0.01

            # Record the start time
            start_time = time.time()
            
            #Call genetic algorithm to mininmize cost function
            # np.random.seed(11)
            Lambda,Pi,Orig,meanPi,minPi,meanParents,costs=GeneticAlgorithmSolarEscape(S,P,K,TOL,G,dv,lb,ub,solarExitCost, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, max_variation, NumSeg, dT, bodies, numSails)
            var = Lambda[0,:]

            print("***************************************************************")
            print("The design variables are:")
            np.set_printoptions(threshold=1000000000)
            print(np.array2string(var, separator=', '))
            print("***************************************************************")
            # print("The corresponding costs are :")
            # print(solarExitCost(var,desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, numSails))
            # print("The minimum cost is: ", costs[0])

            #Plot results
            plt.figure()
            plt.semilogy(np.arange(len(meanPi)), meanPi, label='Mean $\Pi$')
            plt.semilogy(np.arange(len(minPi)), minPi, label='Min $\Pi$')
            plt.semilogy(np.arange(len(meanParents)), meanParents, label='Mean $\Pi_{parents}$')
            plt.title('Cost Evolution per Generation for $\Pi$',fontsize=22)
            plt.xlabel('Generation',fontsize=18)
            plt.ylabel('Value',fontsize=18)
            plt.legend()
            plt.show()
            

    
    else: # Set up experiment to determine best combination of segment lengths, max variations, constant angles, and number of segments

        # Define the arrays
        constant_anglesArray = np.array([0, 1])
        NumSegArray = np.arange(10, 100, 5)
        SegLengthArray = np.arange(10, 140, 40)
        max_variationArray = np.arange(0, 1, 0.15)

        # Define other parameters
        degree_min = 1
        degree_max = 5
        time_min = 3
        time_max = 365
        vel_min = 30 * 1000 / AU  # (AU/s)
        vel_max = 30 * 1000 / AU  # (AU/s)
        setInitDir_min = 0
        setInitDir_max = 1

        # Desired designs and weights
        w = [0] * 6
        w[0] = 1  # Total energy
        w[1] = 10  # Hover Time
        w[2] = 10  # Hover Time 2
        w[3] = 0  # Return
        w[4] = 1  # Closest Distance to Sun
        w[5] = 10  # Approach velocity to NEO

        TOLEarth = 0.1
        TOLNEO = 100000 * 1000 / AU
        # TOLNEO = 0.01

        # Define genetic algorithm parameters
        S = 30
        P = 10
        K = 10
        G = 100
        TOL = 1e-3

        # Record the start time
        start_time = time.time()

        # Initialize variables to store the best cost and design variables
        bestCost = float('inf')
        bestvar = None
        best_experiment = None

        # Calculate the total number of valid experiments
        total_experiments = sum(1 for constant_angles, NumSeg, SegLength, max_variation in itertools.product(constant_anglesArray, NumSegArray, SegLengthArray, max_variationArray) if SegLength >= (1400 / NumSeg))
        print("Total Number of Experiments is:", total_experiments)

        # Iterate over all combinations of the elements in the arrays
        for experiment_index, (constant_angles, NumSeg, SegLength, max_variation) in enumerate(itertools.product(constant_anglesArray, NumSegArray, SegLengthArray, max_variationArray)):

            # Calculate the minimum required segment length based on the number of segments
            min_seg_length = 1400 / NumSeg

            # Skip combinations where the segment length is smaller than 3.5 * NumSeg
            if SegLength < min_seg_length:
                continue
            
            # Calculate the percentage of experiments completed
            percentage_completed = (experiment_index / total_experiments) * 100

            print("                                                               ")
            print("*************************************************************************************************")
            print(f"Experiment {experiment_index + 1}/{total_experiments} ({percentage_completed:.2f}% completed) :")
            print(f"NumSeg = {NumSeg}, SegLength = {SegLength}, constant_angles = {constant_angles}, max_variation = {max_variation}")
            print("*************************************************************************************************")
            print("                                                               ")

            # Set up design variables min and max
            timeSeg_min = np.full(NumSeg, 1)
            timeSeg_max = np.full(NumSeg, SegLength)
            cone_angles_min = np.full(NumSeg, -1.2)
            cone_angles_max = np.full(NumSeg, 1.2)
            clock_angles_min = np.full(NumSeg, -6.2831 / 2)
            clock_angles_max = np.full(NumSeg, 6.2831 / 2)

            # Number of design variables
            dv = 3 + NumSeg * 3

            # Upper and lower bounds for each variable
            lb = np.concatenate([[degree_min, time_min, vel_min], timeSeg_min, cone_angles_min, clock_angles_min])
            ub = np.concatenate([[degree_max, time_max, vel_max], timeSeg_max, cone_angles_max, clock_angles_max])

            

            # Call genetic algorithm to minimize cost function
            # np.random.seed(11)
            Lambda, Pi, Orig, meanPi, minPi, meanParents, costs = GeneticAlgorithm(S, P, K, TOL, G, dv, lb, ub, lightSailCost, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, max_variation, NumSeg, dT, bodies, NEOname = NEOname)
            var = Lambda[0, :]
            # cost = costs[0]

            # Call brute force for fine tuning
            var, cost = bruteForceOptimization(var,desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname = NEOname)

            # Update the best cost and design variables if a better cost is found
            if cost < bestCost:
                bestCost = cost
                bestvar = var
                best_experiment = (NumSeg, SegLength, constant_angles, max_variation)
                print("__________________________________________________________________________________________________________________________")
                print("The best cost SO FAR is : ", bestCost)
                print("The best design variables SO FAR are:")
                np.set_printoptions(threshold=1000000000)
                print(np.array2string(bestvar, separator=', '))
                print("The corresponding cost breakdown is:")
                print(lightSailCost(bestvar, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname=NEOname))
                print(f"The best cost SO FAR happens in experiment with : NumSeg={best_experiment[0]}, SegLength={best_experiment[1]}, constant_angles={best_experiment[2]}, max_variation={best_experiment[3]}")
                print("__________________________________________________________________________________________________________________________")


        print(".......................................................................................................................................")
        print("The best cost is : ", bestCost)
        print("The best design variables are:")
        np.set_printoptions(threshold=1000000000)
        print(np.array2string(bestvar, separator=', '))
        print(f"The best cost happens in experiment with : NumSeg={best_experiment[0]}, SegLength={best_experiment[1]}, constant_angles={best_experiment[2]}, max_variation={best_experiment[3]}")
        print("The corresponding cost breakdown is:")
        print(lightSailCost(bestvar, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname = NEOname))
        print(".......................................................................................................................................")
        

    # Brute force optimization
    if bruteForce==1 and solarSysExit == 0:
        print("***************************************************************")
        print("Brute force optimization starting now....")

        var, cost = bruteForceOptimization(var,desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname = NEOname)

        # var, cost = adamOptimization(var, des, w1, w2, w3, w4, w5, w6, w7, w8, TOLNEO, TOLEarth, dT)

        print("Optimized variables after GA:")
        np.set_printoptions(threshold=1000000000)
        print(np.array2string(Lambda[0,:], separator=', '))
        print("The final cost after GA :")
        print(lightSailCost(Lambda[0,:],desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname = NEOname))

        print("Optimized variables after brute force:")
        print(np.array2string(var, separator=', '))
        print("Final cost after brute force:", cost)
        print(lightSailCost(var,desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, NEOname = NEOname))


    

    if args.filename != None:
        np.savetxt(args.filename, var, delimiter=",")
        print("Saving design variables...")

    # Record the end time
    end_time = time.time()

    # Calculate the total run time
    total_run_time = end_time - start_time

    # Print the total run time
    print(f"Total run time: {total_run_time} seconds")
