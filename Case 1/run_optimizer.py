import numpy as np
import time

from src_python.solver import *
from src_python.genetic_algorithm import *
from src_python.coordinate_descent import *
import src_python.body_data as bd

def lightSailCost(var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies):

    solver = lightSailSolver(var, bodies)
    solver.runSim(desHoverTime, constant_angles, T, TOLNEO, TOLEarth, 0, NumSeg, dT)
    cost = solver.calcCost(w,TOLNEO,TOLEarth)

    return cost

if __name__ == "__main__":

    AU = 1.496e11 # Astronomical Unit (m)

    # Time step
    dT = 5 

    # Total simulation time (years)
    T = 365 * 5

    # Desired hover time in days
    desHoverTime = 80 

    # Choose NEO name (current options are {16 Psyche, Vesta, Eunomia, Bennu, Ceres})
    NEOname = 'Bennu'

    # Option to run coordinate descent
    run_coordinate_descent = 0

    # Random seed
    seed = 12345

    # Set tolerances for NEO and Earth
    TOLEarth=0.1
    # TOLNEO=100000*1000/AU
    TOLNEO=1000/AU

    #Define genetic algorithm parameters
    S=30
    P=10
    K=10
    G=2
    TOL=1e-3

    # Choose whether or not to use constant angles: 0 -> smooth angles, 1 -> constant angles
    constant_angles = 0

    #Design Variables min and max
    NumSeg = 60
    SegLength = 70
    max_variation = 0 # Set to 0 for regular / Set to any other value for "smooth" angle variation
    degree_min = 1
    degree_max = 5
    time_min = 150
    time_max = 365

    #Desired designs and weights
    w = [0] * 6
    w[0] = 1 # Total energy
    w[1] = 10 # Hover Time
    w[2] = 10 # Hover Time 2
    w[3] = 1 # Return
    w[4] = 1 # Closest Distance to Sun
    w[5] = 0 # Approach velocity to NEO
    

    # ------------------------------------------------END OF USER INPUTS------------------------------------
    vel_min=30*1000/AU # (AU/s)
    vel_max=30*1000/AU # (AU/s)
    setInitDir_min = 0
    setInitDir_max = 1
    timeSeg_min=np.full(NumSeg,1)
    timeSeg_max=np.full(NumSeg,SegLength)
    cone_angles_min=np.full(NumSeg,-1.2)
    cone_angles_max=np.full(NumSeg,1.2)
    clock_angles_min=np.full(NumSeg,-6.2831/2)
    clock_angles_max=np.full(NumSeg,6.2831/2)

    #Number of design variables
    dv=3+NumSeg*3

    #Upper and lower bounds for each variable
    lb = np.concatenate([[degree_min, time_min, vel_min],timeSeg_min,cone_angles_min,clock_angles_min])
    ub = np.concatenate([[degree_max, time_max, vel_max],timeSeg_max,cone_angles_max,clock_angles_max])

    # Generate body trajectories
    bodies = bd.get_body_data(NEOname, dT, T)

    # Record the start time
    start_time = time.time()

    #Call genetic algorithm to mininmize cost function
    np.random.seed(11)
    Lambda,Pi,Orig,meanPi,minPi,meanParents,costs=GeneticAlgorithm(S,P,K,TOL,G,dv,lb,ub,lightSailCost, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, max_variation, NumSeg, dT, bodies)
    var = Lambda[0,:]

    print("***************************************************************")
    print("The design variables are:")
    np.set_printoptions(threshold=np.inf)
    print(np.array2string(var, separator=', '))
    print("***************************************************************")
    print("The corresponding costs are :")
    print(lightSailCost(var,desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies))
    print("The minimum cost is: ", costs[0])

    if run_coordinate_descent == 1:
        print("***************************************************************")
        print("Coordinate descent optimization starting now....")

        var, cost = coordinateDescent(var,desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, lightSailCost)

        print("Optimized variables after GA:")
        np.set_printoptions(threshold=1000000000)
        print(np.array2string(Lambda[0,:], separator=', '))
        print("The final cost after GA :")
        print(lightSailCost(Lambda[0,:],desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies))

        print("Optimized variables after brute force:")
        print(np.array2string(var, separator=', '))
        print("Final cost after brute force:", cost)
        print(lightSailCost(var,desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies))


    # Record the end time
    end_time = time.time()

    # Calculate the total run time
    total_run_time = end_time - start_time

    # Print the total run time
    print(f"Total run time: {total_run_time} seconds")



