import numpy as np
import time

from src_python_c2.solver import *
from src_python_c2.genetic_algorithm import *
from src_python_c2.coordinate_descent import *
from src_python_c2.solarExitCost import *
import src_python_c2.body_data as bd
import src_python_c2.visuals as vis


def optimizer_input_data():
    # Time step
    dT = 5 

    # Total simulation time (years)
    T = 365 * 5

    # Number of sails in the swarm
    numSails = 5

    # Number of angle segments and maximum lengths
    NumSeg = 30
    SegLength = 70

    #Desired designs and weights
    w = [0] * 4
    w[0] = 1 # Closest distance to Sun penalty
    w[1] = 1 # Distance from Sun
    w[2] = 1 # Distances between sails
    w[3] = 10 # Percent Success

    # Choose whether or not to use constant angles: 0 -> smooth angles, 1 -> constant angles
    constant_angles = 0

    return dT, T, numSails, NumSeg, SegLength, constant_angles, w




if __name__ == "__main__":

    AU = 1.496e11 # Astronomical Unit (m)

    dT, T, numSails, NumSeg, SegLength, constant_angles, w = optimizer_input_data()
    
    # Option to run coordinate descent
    run_coordinate_descent = 0

    # Random seed
    seed = 12345

    #Define genetic algorithm parameters
    S=30
    P=10
    K=10
    G=2
    TOL=1e-3

    

    #Design Variables min and max
    max_variation = 0 # Set to 0 for regular / Set to any other value for "smooth" angle variation
    degree_min = 1
    degree_max = 5
    time_min = 150
    time_max = 365

    
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
    bodies = bd.get_body_data(dT, T)

    # Record the start time
    start_time = time.time()

    #Call genetic algorithm to mininmize cost function
    np.random.seed(seed)
    Lambda,Pi,Orig,meanPi,minPi,meanParents,costs=GeneticAlgorithmSolarEscape(S,P,K,TOL,G,dv,lb,ub, solarExitCost, constant_angles, T, w, max_variation, NumSeg, dT, bodies, numSails)
    var = Lambda[0,:]

    if run_coordinate_descent == 1:
        print("***************************************************************")
        print("Coordinate descent optimization starting now....")

        var, cost = coordinate_descent_SolarEscape(var, constant_angles, T, w, dT, bodies, numSails, dv, NumSeg, lb, ub, solarExitCost)
        store_design_variables(f'solar_exit_design_variables_CD_{numSails}_Sails.txt', var, numSails, NumSeg, dT, T, SegLength)
    else:
        # Store the design variables from the GA only:
        store_design_variables(f'solar_exit_design_variables_GA_{numSails}_Sails.txt', var, numSails, NumSeg, dT, T, SegLength)

    solarExitCost(var, constant_angles, T, w, dT, bodies, numSails, dv, NumSeg)

    #Plot results
    vis.make_plots(meanPi, minPi, meanParents)

    # Record the end time
    end_time = time.time()

    # Calculate the total run time
    total_run_time = end_time - start_time

    # Print the total run time
    print(f"Total run time: {total_run_time} seconds")



