import numpy as np
import os

from .solver import *

def store_design_variables(filename, var, numSails, NumSeg, dT, T, SegLength):
    # Save the animation
    output_dir = "Case 2/output"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    
    with open(full_path, 'w') as f:
        f.write(f'Design Variables for numSails = {numSails}, dT = {dT}, T = {T/365} Years, NumSeg = {NumSeg},  and SegLength = {SegLength}:\n')
        np.set_printoptions(threshold=np.inf)  # Ensure the entire array is printed
        f.write(np.array2string(var, separator=', '))


def solarExitCost(var, constant_angles, T, w, dT, bodies, numSails, dv, NumSeg):
    finalPositions = []
    sunCosts = []
    distCosts = []
    posCosts = []

    for i in range(numSails):
        solver = lightSailSolver(var[i*dv:(i+1)*dv], bodies)
        solver.runSim(constant_angles, T, NumSeg, dT)
        cost = solver.calcCostSolarExit()

        sunCosts.append(cost[0])
        distCosts.append(cost[1])
        finalPositions.append(cost[2])
        
    finalPositions = np.array(finalPositions)
    count = 0

    # Calculate pairwise distances and find the smallest distance for each sail
    for i in range(numSails):
        distances = np.linalg.norm(finalPositions[i] - finalPositions, axis=1)
        distances[i] = np.inf
        min_distance = np.min(distances)
        posCosts.append(min_distance)

        if distCosts[i] < 0.8:
            count+=1
        
        if distCosts[i] > 1:
            distCosts[i] = 0
    
    percentSuccess = count/numSails 
    auxCost = 1 - percentSuccess

    sumDistCost = np.sum(distCosts)

    finalCost = w[0] * np.max(sunCosts) + w[1] * np.max(distCosts) - w[2] * np.min(posCosts) + w[3] * auxCost + sumDistCost

    print("Cost contribution 1 (Closest Distance to Sun):", w[0] * np.max(sunCosts))
    print("Cost contribution 2 (Distance away from Sun):", w[1] * np.max(distCosts))
    print("Cost contribution 3 (Distances from each other):", - w[2] * np.min(posCosts))
    print("Cost contribution 4 (Percent Success):", w[3] * auxCost)
    print("Cost contribution 5 (Sum Distances):", sumDistCost)
    print("Total cost: ", finalCost)

    return finalCost