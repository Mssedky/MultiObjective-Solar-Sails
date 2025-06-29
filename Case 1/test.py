import time
import numpy as np

from src_python.solver import *
import src_python.body_data as bd
import src_python.visuals as vis

vars = [ 3.00000000e+00,  3.32128464e+02,  2.00534759e-07,  5.50493134e+01,
  5.60007259e+01,  1.16276188e+00,  6.36755791e+01,  3.08946961e+00,
  5.69572916e+00,  4.51337035e+01,  1.06946799e+01,  5.33738252e+01,
  6.90699695e+01,  5.11019923e+01,  1.02820670e+01,  2.02679082e+01,
  5.53985054e+01,  4.28350490e+01,  3.47482226e+01,  1.99932930e+01,
  1.24465194e+01,  2.36854308e+01,  2.65079602e+01,  2.18393099e+01,
  5.50972267e+01,  1.46893366e+01,  1.93124779e+01,  3.64997516e+01,
  3.16810450e+01,  3.19302475e+01,  3.92753670e+00,  3.30354319e+01,
  5.68020026e+01,  6.33930295e+01,  5.52060438e+01,  2.50255059e+01,
  4.52512732e+01,  2.64493699e+01,  3.31169699e+01,  6.66382802e+01,
  3.62850130e+01,  4.71601165e+01,  3.29768410e+01,  4.45501959e+01,
  4.00285654e+01,  3.06180788e+00,  1.77533119e+01,  5.01218400e+01,
  3.68873443e+01,  5.57740960e+01,  1.22397373e+01,  4.67988612e+00,
  4.71750324e+01,  4.20358689e+01,  4.36301062e+01,  5.58102924e+01,
  1.76909107e+01,  6.54666186e+01,  6.21336812e+01,  6.33631853e+01,
  5.48295705e+01,  1.92812202e+01,  5.99887934e+01,  1.07653477e+00,
  1.09463604e+00,  9.36472292e-01,  6.50947035e-01, -6.71972159e-01,
 -8.21244442e-01,  1.03207272e-01,  9.53121635e-01,  5.18124178e-02,
  5.72087458e-02,  4.97133999e-01, -9.07792403e-01, -1.20617988e-01,
 -3.85110241e-01, -5.60164410e-01, -6.49469820e-01,  7.48625666e-01,
  1.12984327e+00,  6.26669514e-01,  1.08347115e+00,  4.77600339e-02,
 -3.66956485e-01,  1.09045458e+00, -5.42269724e-01, -9.93633496e-01,
  1.87729459e-01,  9.60053808e-01, -7.85685722e-02, -2.52452383e-01,
  3.04886332e-01, -2.17683682e-01,  2.00991320e-01, -4.18772508e-01,
 -8.72307866e-01,  2.16789965e-01,  5.56899466e-01,  5.56989467e-01,
  2.07878447e-01, -1.01178896e+00,  4.90215003e-01, -6.35949467e-01,
 -1.11761591e+00, -1.04303157e-01,  6.79165833e-01, -2.95708084e-01,
 -2.17110703e-01,  4.90161441e-01,  4.03071327e-02,  6.00032788e-02,
  1.04177855e+00, -3.63342160e-01,  2.93285987e-01, -8.36050538e-01,
  6.56059957e-01, -3.24258829e-01,  5.27168578e-01,  2.40136779e-01,
  9.81681568e-01, -9.40569998e-01, -9.83281054e-01,  2.78221092e+00,
  4.83440078e-02,  1.84554884e+00, -1.73900847e-01, -2.82707502e+00,
  1.66370571e+00, -9.19720528e-01, -8.94468488e-02,  7.99586955e-01,
  8.67384819e-01,  1.69438068e+00,  6.13543324e-01, -1.80038773e+00,
  1.47956659e+00, -2.39979955e+00,  2.31898766e+00,  3.01494428e+00,
  1.89681605e+00, -3.68325875e-01, -7.21359712e-01,  2.44810097e+00,
 -3.01230152e+00,  2.50097999e+00, -1.41345167e+00, -8.13707191e-01,
 -1.27476640e+00, -6.55297690e-01,  1.40143335e+00,  3.11817567e+00,
 -5.88417484e-01, -1.71147850e+00, -1.31948680e+00,  6.01909989e-01,
  1.24414331e+00,  1.13743607e+00,  6.11478186e-01,  1.28645201e+00,
  9.21366625e-01,  2.31890478e+00,  9.66248499e-01,  3.03492109e+00,
  9.02838214e-01,  1.61490316e+00, -2.77712412e+00,  1.87422429e+00,
  1.68240148e+00, -4.05838204e-01, -2.14456185e+00, -2.36477958e+00,
 -1.33699926e+00,  1.37696153e+00, -3.90160383e-02,  2.29965523e-01,
  2.50120738e+00,  1.81594363e+00,  4.23358907e-01, -3.07613179e+00,
 -2.73851628e+00,  6.06118481e-01, -1.97822408e+00]


if __name__ == "__main__":

    AU = 1.496e11 # Astronomical Unit (m)

    # Time step
    dT = 5 

    # Total simulation time (years)
    T = 365 * 5

    # Choose whether or not to use constant angles: 0 -> smooth angles, 1 -> constant angles
    constant_angles = 0

    # Choose whether or not to save trajectory data
    savetraj = 1

    # Choose whether or not to plot
    plots = 1

    # Choose whether or not to create animation
    movie = 1

    # Desired hover time in days
    desHoverTime = 80 

    # Choose NEO name (current options are {16 Psyche, Vesta, Eunomia, Bennu, Ceres})
    NEOname = 'Bennu'

    NumSeg = round((np.array(vars).shape[0]-3)/3)

    # Set tolerances for NEO and Earth
    TOLEarth=0.1
    # TOLNEO=100000*1000/AU
    TOLNEO=1000/AU    

    #Desired designs and weights
    w = [0] * 6
    w[0] = 1 # Total energy
    w[1] = 10 # Hover Time
    w[2] = 10 # Hover Time 2
    w[3] = 1 # Return
    w[4] = 1 # Closest Distance to Sun
    w[5] = 0 # Approach velocity to NEO


    # Generate body trajectories
    bodies = bd.get_body_data(NEOname, dT, T)

    # Record the start time
    start_time = time.time()

    solver = lightSailSolver(vars,bodies)

    solver.runSim(desHoverTime, constant_angles, T, TOLNEO, TOLEarth, savetraj, NumSeg, dT, \
                traj_output_filename = f'sail_trajectory_{NEOname}_python', trackNEO = 1, useEndCond = 1, NEOname = NEOname)
    
    cost = solver.calcCost(w,TOLNEO,TOLEarth)

    sailPos = solver.sailPos
    earthPos = solver.earthPos
    NEOPos = solver.NEOPos
    sunPos = solver.sunPos
    distances = solver.distances
    simTime = solver.simTime
    ToF = solver.ToF
    alphaG = solver.alphaG
    gammaG = solver.gammaG

    
    # Find the index of the smallest distance
    min_distance_index = np.argmin(distances)

    minDist=np.min(distances)

    # Get the corresponding time step
    min_distance_time = ToF[min_distance_index]



    ###############################################################################################
    #Plots
    if plots==1:
        vis.make_plots(ToF, alphaG, gammaG, distances)

    #Plotting sunPos, earthPos, NEOPos, sailPos, and simTime in a movie:
    if movie==1:
        vis.make_animation(sunPos, earthPos, NEOPos, sailPos,simTime, NEOname)

   


