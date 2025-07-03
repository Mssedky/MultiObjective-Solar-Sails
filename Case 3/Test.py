import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fsolve
from scipy.optimize import root
import argparse
import argparse

from LightSailSolver import *
from BodiesCalc import *

def export_to_tecplot_SolarExit(filename, time_steps, sun_data, light_sail_data, sail_normal, sailFlightPos, planet_data, num_sails):
    with open(filename, 'w') as f:
        # Write title and variable definitions
        f.write('TITLE = "Light Sail Trajectory"\n')
        f.write('VARIABLES = "X", "Y", "Z", "NX", "NY", "NZ"\n')
        
        # Loop through time steps and write zones for each entity
        for i, time in enumerate(time_steps):
            # Write zone for Sun
            f.write(f'ZONE T="Sun", SOLUTIONTIME={time}\n')
            f.write(f'{sun_data[i,0]} {sun_data[i,1]} {sun_data[i,2]} 0 0 0\n')
                
            # Write zones for Light Sails
            for sail_idx in range(num_sails):
                f.write(f'ZONE T="Light Sail {sail_idx+1}", SOLUTIONTIME={time}\n')  
                f.write(f'{light_sail_data[sail_idx][i,0]} {light_sail_data[sail_idx][i,1]} {light_sail_data[sail_idx][i,2]} {sail_normal[sail_idx][i,0]} {sail_normal[sail_idx][i,1]} {sail_normal[sail_idx][i,2]}\n')
            
            # Write zones for Planets
            planet_names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
            for planet_idx, planet_name in enumerate(planet_names):
                f.write(f'ZONE T="{planet_name}", SOLUTIONTIME={time}\n')
                f.write(f'{planet_data[planet_idx][i,0]} {planet_data[planet_idx][i,1]} {planet_data[planet_idx][i,2]} 0 0 0\n')
        
        f.write(f'ZONE T="Light Sail Trajectory"\n')
        for sail_idx in range(num_sails):
            for i in range(len(sailFlightPos[sail_idx])):
                f.write(f'{sailFlightPos[sail_idx][i,0]} {sailFlightPos[sail_idx][i,1]} {sailFlightPos[sail_idx][i,2]} 0 0 0\n')
        
        # Write zones for Planet Trajectories
        for planet_idx, planet_name in enumerate(planet_names):
            f.write(f'ZONE T="{planet_name} Trajectory"\n')
            for i in range(len(planet_data[planet_idx])):
                f.write(f'{planet_data[planet_idx][i,0]} {planet_data[planet_idx][i,1]} {planet_data[planet_idx][i,2]} 0 0 0\n')

dT_default = 5 #default time step = 5 days
plots_default = 1 #default: produce plots
movie_default = 1 #default: produce movie
savetraj_default = 0 #default: save trajectory data
trackNEO_default = 1 #default: track NEO
useEndCond_default = 1 #default: use end condition
solarSysExit_default = 0 #default: don't use solar sys exit version
NEOname_default = 'Vesta' #default: target NEO object is Vesta
T_default = 365*10 #default: max simulation time is 10 years

parser = argparse.ArgumentParser(
                    prog='LightSail_GA.py',
                    description='Genetic algorithm for single trajectory optimization')
parser.add_argument('--inputfilename')       # filename for input variables in form of csv
parser.add_argument('--savetraj')            # save trajectory data if 1
parser.add_argument('--outputfilename')       # filename (without file extension) for saving final trajectory. will save two files (filename.dat, filename.csv).  if no filename given and savetraj=0, movie = 0, no save. if no filename given and savetraj=1 and/or movie = 1, saved under default name.
parser.add_argument('-dT', '--dT')            # time step size (days)
parser.add_argument('--plots')            # generate plots if 1
parser.add_argument('--movie')            # generate movie if 1
parser.add_argument('--trackNEO')            # track NEO during simulation if =1
parser.add_argument('--useEndCond')          # end simulation when it reaches end condition if =1
parser.add_argument('--solarSysExit')          # run solar system exit version of code if =1
parser.add_argument('--NEOname')          # Name of target object for simulation. options: 'NEO', '16 Psyche', 'Vesta', 'Eunomia', 'Ceres'
parser.add_argument('--T')          # maximum simulation time (days)

args = parser.parse_args()

dT = args.dT if args.dT!=None else dT_default
plots = int(args.plots) if args.plots!=None else plots_default
movie = int(args.movie) if args.movie!=None else movie_default
if args.savetraj == None:
    savetraj = savetraj_default
elif int(args.savetraj) == 1 or movie == 1:
    savetraj = 1
else:
    savetraj = int(args.savetraj)
trackNEO = int(args.trackNEO) if args.trackNEO!=None else trackNEO_default
useEndCond = int(args.useEndCond) if args.useEndCond!=None else useEndCond_default
solarSysExit = int(args.solarSysExit) if args.solarSysExit!=None else solarSysExit_default
NEOname = args.NEOname if args.NEOname!=None else NEOname_default
T = float(args.T) if args.T!=None else T_default

# Record the start time
start_time = time.time()

#Simulation (Optimum) Parameters (First is best so far)
# var_default = [3.27082211e+02, 1.60427807e-07, 2.33231094e+01, 4.26882957e+01,
# var_default = [3.27082211e+02, 1.60427807e-07, 2.33231094e+01, 4.26882957e+01,
#  4.15585192e+01, 5.56817631e+01, 6.88953345e+00, 5.22494090e+01,
#  7.21790791e+01, 1.34392615e+01, 3.10527231e+01, 3.61195392e+01,
#  6.38277358e+01, 1.05640447e+01, 2.80509792e+00, 2.62067385e+01,
#  2.07070835e+01, 2.13395107e+01, 3.35041703e+00, 4.79489289e+01,
#  1.24733782e+00, 1.22118131e+02, 2.94223167e+00, 2.74543163e+00,
#  5.13283440e+00, 2.41183346e+00, 3.05898756e+00, 1.69018797e+00,
#  5.16860106e+00, 5.04048531e+00, 3.81101120e+00, 3.00896606e-01,
#  2.77799627e+00, 3.11549256e+00, 2.58370786e+00, 6.09471660e+00,
#  2.18029346e+00, 3.82571359e+00, 5.41354232e+00, 2.46436376e+00,
#  2.87031381e-01, 3.98026533e-01, 4.67949506e+00, 2.06448146e+00,
#  2.70254399e+00, 1.89709329e+00, 2.10847373e+00, 5.99662733e+00,
#  4.16532769e+00, 5.51820832e+00, 3.93930571e+00, 5.00392182e+00,
#  5.37309398e+00, 1.68382325e+00, 5.99955314e+00, 4.59394639e-01,
#  8.71515820e-01, 4.49730053e+00, 4.73163523e+00, 1.54739524e+00,
#  2.56415184e+00, 1.70458139e+00]


# var_default = [ 4.00000000e+00,  2.23235416e+02,  2.00534759e-07,  2.11341426e+01,
#   3.37416434e+01,  4.91216228e+01,  4.72478203e+01,  4.23362693e+01,
#   2.97162161e+01,  2.57439163e+01,  4.22181946e+01,  2.47310516e+01,
#   4.95996728e+01,  4.01552458e+01,  4.62335338e+01,  4.17863047e+01,
#   5.97169170e+01,  5.73315239e+01,  2.18222937e+01,  3.39962782e+01,
#   2.40863455e+01,  1.99146277e+01,  5.93789780e+01,  2.61979880e+01,
#   1.93296802e+01,  3.60763891e+01,  4.30704224e+01,  1.07375997e+01,
#   4.69610642e+01,  1.85215200e+01,  2.03633843e+01,  2.86323414e+01,
#   2.02797171e+01, -1.22760499e-01,  1.15340917e-01, -7.24604189e-01,
#  -4.74799415e-02,  2.10694861e-01, -1.92460976e-01, -6.77014941e-01,
#   8.83093223e-01, -8.40608940e-01,  5.48619822e-01,  7.39841206e-01,
#   4.45284783e-02, -4.88396533e-01,  2.10860424e-01,  2.02605731e-01,
#  -9.73833478e-02,  9.16682380e-01, -2.87872688e-01,  4.72111835e-01,
#   3.70586347e-01, -9.40274355e-02,  6.72565681e-01, -6.49252628e-01,
#  -5.06494827e-02, -1.51044428e-02,  3.10579081e-01,  2.44159866e-01,
#   2.32562138e-01,  2.63222564e-01, -6.62901331e-01, -2.36424278e+00,
#  -2.04961947e-02, -2.08454989e+00, -1.52320099e+00, -1.10867067e-02,
#   2.09556244e+00, -3.84475763e-01,  2.60765046e+00, -9.87190548e-01,
#   7.30703273e-01,  1.25404248e-01,  5.89370872e-01, -2.82875729e-01,
#   3.73592516e-01, -2.19462089e+00,  6.43942157e-01, -2.48892092e+00,
#   2.14954909e-01,  3.58486566e-01, -2.35307430e-01, -2.32281887e+00,
#  -1.14468737e+00,  9.34538195e-01, -8.15003682e-01,  9.73354174e-01,
#  -1.85345727e-01,  4.70950325e-01, -3.18133803e+00, -1.07526451e+00,
#  -1.25474741e+00]

# Vesta rendezvous for T= 4 years, dT = 0.1
# Optimized variables after brute force:
# [ 5.00000000e+00,  6.91670173e+01,  2.00534759e-07,  7.27117853e+01,
#   7.07431813e+01,  2.02449682e+01,  5.51272396e+01,  2.01484714e+01,
#   2.92135189e+01,  4.39354974e+01,  3.97858043e+01,  4.55839286e+01,
#   4.95367915e+01,  7.34228353e+00,  2.04278736e+01,  6.73081217e+01,
#   3.88381683e+00,  4.96856371e+01,  5.40516151e+01,  1.37363954e+01,
#   2.43123384e+01,  6.39666084e+01,  1.84485266e+01,  2.45421881e+00,
#   5.53986453e+01,  9.72033771e+00,  4.31467311e+01,  6.66510982e+01,
#   1.79093322e+01,  9.49377726e+00,  1.25347092e+01,  3.73069792e+01,
#   5.37859432e+01,  5.56443969e-01, -1.71319061e-01, -2.30445403e-01,
#  -2.12143968e-01, -5.63719141e-02, -5.87746057e-01, -9.75020019e-01,
#  -7.20988933e-01, -5.51834395e-01, -9.33040956e-01,  8.86504093e-01,
#  -8.90814513e-01, -9.49711121e-01, -1.33894606e-01, -4.87876605e-01,
#  -7.25214032e-01,  7.90236745e-01, -4.97037429e-01,  6.27465944e-01,
#   5.56338077e-01,  3.94816770e-01,  7.97568197e-01, -9.50911525e-01,
#   5.94915806e-01,  1.19341287e+00, -1.14151950e+00, -1.40445012e-01,
#   9.67098161e-01, -2.78084426e-01, -3.45206353e-02, -4.48342340e-01,
#   4.05737785e+00, -2.62614892e+00,  8.51267610e-02,  2.23459369e+00,
#   9.97270760e-01,  2.02855883e+00,  6.04874910e-01,  3.08287929e+00,
#   8.82593281e-01,  2.03082763e+00, -9.93153053e-01,  2.68235479e+00,
#   2.08047963e+00,  1.87734998e+00,  2.06116128e+00, -1.09921380e+00,
#  -2.33402877e+00, -3.99671000e-01, -2.20320653e+00,  2.18233861e-01,
#   7.00485841e-01,  1.63208823e+00,  2.08828887e+00, -5.12831345e-01,
#  -1.66886811e+00,  1.95659905e+00,  1.90891167e+00,  3.60824922e+00,
#   3.83743522e-01]
# Final cost after brute force: 14.048930069869662
# Shortest time to NEO: 1194.3000000000093 days
# Total flight time: 1459.9999999997676 days
# Returned to earth: 0
# Cost contribution 1 (Total energy): 0.7792554496860918
# Cost contribution 2 (Hover Time): 0.995
# Cost contribution 3 (Hover Time 2): 8.608333333333334
# Cost contribution 4 (Return): 3.6663412868502374
# Cost contribution 5 (Closest Distance to Sun): 0
# Cost contribution 6 (Approach velocity to NEO): 0
# Total cost:  14.048930069869662
# 14.048930069869662
# Total run time: 21138.5588388443 seconds



# Solar system escape for 15 years, 0.1 TOL sunClose, and 30 sails
# The design variables are:
var_default = [ 5.00000000e+00,  2.61631367e+02,  2.00534759e-07,  6.23425087e+01,
  1.90824530e+01,  5.04247856e+01,  1.44249142e+01,  5.18818603e+01,
  6.79352666e+01,  3.24876985e+01,  1.35327713e+01,  1.12508241e+01,
  7.27182377e+01,  2.36293279e+01,  7.63545203e+01,  6.53543059e+01,
  6.17572758e+01,  5.97299094e+01,  3.06802236e+01,  4.41399136e+01,
  3.43136080e+01,  6.56076610e+01,  7.30932202e+01,  5.53445133e+00,
  4.58401433e+01,  3.03918968e+01,  7.24494917e+01,  1.08916965e+01,
  6.02224780e+01,  8.35104456e+00,  5.49439835e+01,  4.30135175e+01,
  6.57944122e+01, -1.11424810e+00,  7.04765605e-01,  1.11782884e+00,
  5.58012377e-01,  1.12062083e+00,  1.02588338e+00, -4.81598589e-01,
 -8.90578730e-02, -1.04961858e+00,  1.13628060e+00, -4.90193874e-01,
  6.32184650e-01,  1.06822429e+00,  2.74159090e-01,  1.21220218e-01,
  6.56091027e-02,  5.67592422e-01, -1.01305710e+00, -4.10518896e-01,
 -9.18716831e-01,  5.22848181e-01,  9.70478704e-02, -6.75238485e-01,
 -4.76929438e-01, -5.76308861e-01, -1.21706772e-01,  7.07366132e-01,
 -3.91635465e-01,  1.06518194e+00,  4.63904833e-01, -1.68478848e+00,
  1.37046640e+00, -2.33639332e-01,  1.79718649e-01, -1.48561037e+00,
  2.34732156e+00, -1.07223473e+00,  3.03040870e+00, -1.10882132e+00,
 -1.76281323e+00, -2.35809579e+00, -1.70591362e+00,  8.30249008e-01,
 -2.38349258e+00, -2.12737875e+00, -3.35658459e-01, -1.68654872e+00,
  9.61648254e-01,  1.86961347e+00, -5.54201618e-01, -3.92395832e-01,
 -2.58049033e-01,  2.68459434e+00, -2.66582817e+00,  1.31178711e+00,
 -1.10367404e+00, -4.20579999e-01,  2.50501753e+00, -2.64135113e+00,
  3.86203895e-01,  5.00000000e+00,  3.03817410e+02,  2.00534759e-07,
  1.99124532e+01,  3.76178030e+01,  4.14011144e+01,  3.31245744e+01,
  4.85271547e+01,  1.06347802e+00,  2.95984414e+01,  6.91520057e+01,
  1.06811020e+01,  7.71782708e+01,  2.26916527e+01,  1.80077403e+01,
  4.38272130e+01,  6.95015462e+01,  5.05031036e+01,  2.19036615e+01,
  5.37848123e+00,  1.10376820e+01,  6.45512436e+01,  5.31047971e+01,
  4.16229788e+01,  6.46613815e+01,  4.99350792e+01,  5.73163516e+01,
  5.80764527e+01,  7.74606302e+01,  5.15313332e+01,  7.03695166e+01,
  6.31932455e+01,  4.79169757e+01, -1.13824166e+00,  7.16127373e-01,
 -2.42265282e-02, -1.94809346e-02,  6.24689955e-02,  5.72059392e-01,
 -8.59671879e-01,  1.03479333e+00, -1.91246341e-01,  7.53942760e-02,
  9.35417870e-01, -6.48132952e-01, -1.04287190e+00, -1.00852970e+00,
  5.67320708e-01, -1.19896175e+00, -1.57686095e-01, -1.03831599e+00,
 -6.09277597e-01,  1.17133202e+00, -9.31109769e-01,  1.68448351e-01,
 -5.78233819e-01,  7.19582353e-01,  2.50977989e-01, -3.40792308e-01,
 -1.19131520e+00, -9.34658943e-02,  1.02203386e+00, -6.63478359e-01,
  2.10651742e+00,  9.96552842e-01,  2.01039813e+00, -3.04796817e+00,
  4.76600736e-01, -1.46446631e+00,  1.70256283e+00, -1.59448287e+00,
  2.88849584e+00, -1.33974523e+00, -7.95665106e-01,  1.81178960e+00,
  1.55993243e+00,  2.16354032e+00,  2.17052323e-01, -9.45158355e-01,
 -1.33600002e+00,  1.34584633e+00,  1.34290163e+00, -1.37137438e+00,
 -5.60970935e-01, -2.00318894e+00,  1.44495880e+00, -1.12127481e+00,
 -2.49541854e+00,  1.32223999e+00,  2.53571398e+00, -1.39555431e+00,
  1.45475412e+00,  5.73429386e-01,  2.00000000e+00,  3.55887538e+02,
  2.00534759e-07,  1.12607185e+01,  3.40923845e+01,  5.87746532e+01,
  3.33291759e+01,  1.23669090e+01,  5.69955888e+01,  6.84132180e+01,
  1.08350552e+01,  1.04629042e+01,  5.26846307e+01,  5.69313504e+01,
  4.34970001e+01,  4.91258502e+01,  2.67630985e+01,  1.70727517e+01,
  6.84698606e+01,  1.48778086e+01,  6.72354464e+01,  7.32003384e+01,
  3.79331025e+00,  3.16226527e+00,  5.86495900e+01,  3.07021969e+00,
  4.73287221e+01,  7.30186929e+01,  1.55129190e+01,  3.49763333e+01,
  6.23151398e+00,  2.43813412e+01,  2.76608635e+01,  2.52929264e-01,
 -1.85619710e-01, -2.50528210e-01,  1.00374518e+00, -4.06597397e-01,
  1.93112772e-01,  1.66000156e-01, -1.09000398e+00,  5.37980189e-01,
  8.87374453e-02,  9.18659404e-01, -4.71024034e-01, -2.07341427e-01,
  1.12231972e+00,  3.66115192e-01, -7.05633350e-01,  4.88399011e-02,
 -4.79392607e-02, -6.54967880e-01,  4.49436206e-01,  9.93292888e-01,
  1.29959582e-01, -6.50831951e-01, -6.94638847e-01,  2.63181364e-01,
 -7.98106906e-01, -1.12346453e-01, -5.71830905e-01, -7.47451675e-01,
 -6.79780230e-01,  2.03976560e+00, -1.54612600e+00, -2.84996946e+00,
  1.80243712e+00, -2.41472992e+00,  7.73472430e-01, -1.16669026e+00,
 -2.47912523e+00, -7.15563541e-01,  2.29123422e+00, -2.98699252e+00,
 -8.72517635e-01,  8.52696383e-02,  2.09291632e+00,  7.36080935e-01,
 -8.01613558e-01,  2.65698730e+00,  7.64876594e-01,  1.23515480e-01,
  1.03952135e+00, -3.48824025e-01, -3.02129530e+00,  2.25874973e+00,
 -5.71592922e-01,  1.18697919e+00, -9.82728608e-01, -2.50961089e+00,
  1.40857670e+00,  2.30476038e-01,  1.95599529e-01,  1.00000000e+00,
  3.32953888e+02,  2.00534759e-07,  4.14207912e+01,  5.85917942e+01,
  1.62319569e+01,  3.64857699e+01,  2.73770628e+01,  6.18053613e+01,
  4.71093947e+01,  1.07995032e+01,  5.73849682e+01,  7.81196138e+00,
  7.55696875e+01,  7.78574835e+01,  7.69244999e+01,  1.64701518e+01,
  6.01002454e+01,  7.72492455e+01,  3.95977394e+01,  3.46993959e+01,
  4.56012035e+01,  6.44397495e+01,  2.69660642e+01,  2.39231539e+01,
  5.30666123e+01,  5.13908899e+01,  7.98887173e+01,  3.61195315e+01,
  5.27920140e+01,  6.24031610e+00,  3.16297716e+01,  6.38913828e+01,
  1.85507093e-01,  9.25753428e-01,  1.02031334e+00,  4.29757225e-01,
 -1.10586063e+00, -9.12805056e-01,  1.18633217e+00,  1.10577890e+00,
  5.72851740e-01, -5.86313011e-02,  4.63647128e-01,  6.68509820e-01,
 -5.96092563e-01,  1.06158498e+00, -3.43176533e-01, -9.79682885e-01,
 -7.60357103e-01, -7.50311520e-01, -2.39305181e-01,  3.65008131e-02,
  7.45410981e-01, -6.53512582e-01, -8.78117117e-01, -2.75925325e-01,
 -3.56077114e-01,  8.72372098e-01,  5.13443129e-01,  8.37626791e-01,
  6.54136081e-01,  5.59240330e-02,  5.80798736e-01, -2.17715630e+00,
 -7.10123774e-02,  1.13868528e+00,  1.92147933e+00, -2.43262057e+00,
 -3.67324041e-01, -1.67801082e-01, -9.62275421e-01,  2.53092763e+00,
 -2.49869451e+00,  7.80805758e-01,  1.57111627e+00,  1.21213008e+00,
 -6.52948580e-01,  1.50589531e+00, -2.02696961e+00, -2.98397394e+00,
  2.07582357e+00, -3.05341569e+00, -2.07666533e+00,  1.38235753e+00,
  2.56830913e+00, -9.02379941e-01, -5.60633287e-01,  2.04289239e+00,
 -2.60507546e+00,  7.68418897e-01, -2.88510279e+00,  1.19075145e+00,
  1.00000000e+00,  2.04783705e+02,  2.00534759e-07,  5.08096228e+01,
  3.02207411e+01,  1.54005841e+00,  5.99792886e+00,  1.23913249e+01,
  2.09460320e+01,  1.69857854e+01,  7.12535257e+01,  1.36182528e+01,
  5.75408085e+01,  3.78633141e+01,  1.97897159e+01,  7.19848030e+01,
  3.96868347e+01,  3.33395480e+01,  5.15082205e+01,  4.72941203e+01,
  2.74442262e+01,  5.26881645e+01,  1.67205514e+01,  4.79120664e+01,
  5.79712906e+01,  6.82709911e+01,  1.48079601e+01,  4.57902036e+01,
  4.95211539e+01,  1.35940616e+01,  7.49957398e+01,  4.45271834e+01,
  6.81748365e+01, -5.07627520e-01, -5.99079465e-01,  6.37028049e-01,
 -7.62587733e-01, -2.51794365e-01,  8.34164840e-01,  2.87218269e-02,
 -1.17663762e+00,  2.14621161e-02,  6.10704466e-01, -9.22809988e-01,
  1.14983693e+00,  7.05172689e-01, -1.15235471e+00, -4.04681409e-01,
 -1.12167368e-01,  1.19640375e+00, -1.86293796e-01, -7.12016667e-01,
 -7.62641156e-01,  8.63267595e-01, -2.98023498e-01,  8.22217572e-01,
  2.08791783e-01, -8.00128071e-02, -5.85203428e-01,  8.46838173e-01,
 -4.78358808e-01, -4.99240357e-01, -1.18815401e+00, -6.15017915e-01,
  9.26919773e-01, -4.57183005e-01, -7.28635567e-01,  1.31377140e+00,
  2.59075513e+00, -1.08462867e-01, -1.59209395e+00, -1.58985167e+00,
 -5.95180443e-01, -2.49430652e+00,  9.17200452e-01, -2.70446734e+00,
 -1.51727110e-01, -2.23363653e+00,  1.10454434e+00, -2.36182496e+00,
  3.84636799e-01, -2.82631449e+00, -9.60395215e-01,  1.31365628e+00,
  2.53350664e+00, -4.60498692e-01,  8.74072582e-01, -7.82137400e-01,
  2.74999628e+00,  1.10187867e+00, -1.80755150e-01,  1.15890838e+00,
  7.21480827e-01,  4.00000000e+00,  7.27919258e+01,  2.00534759e-07,
  1.40437983e+01,  1.34328313e+00,  1.20429088e+01,  7.76444081e+01,
  1.34118100e+00,  1.03493770e+01,  6.21281737e+01,  3.26995687e+01,
  2.73853595e+01,  2.41383768e+01,  1.05013155e+01,  2.11793786e+01,
  5.10052702e+01,  3.31684876e+01,  6.77658615e+01,  7.06390542e+01,
  2.96936320e+01,  7.05853745e+01,  3.15930861e+01,  5.27035424e+01,
  7.08305793e+01,  3.34826835e+01,  5.25493376e+00,  5.39102911e+01,
  6.52067301e+01,  3.62467684e+01,  3.13999788e+01,  5.16831254e+01,
  5.90613769e+01,  6.59935485e+01,  4.00787829e-01,  5.30618956e-01,
  4.33673880e-01, -1.14488949e+00,  2.52763586e-01,  1.93381888e-01,
  5.75502559e-01, -1.13955276e+00, -6.16235491e-01,  7.38160593e-01,
  8.92654557e-01,  4.29958149e-01,  9.03240012e-01,  2.35566199e-03,
  1.91617622e-01,  1.14075762e+00, -6.19219129e-01, -7.39798061e-01,
 -8.21304655e-01, -5.15888155e-01, -1.48388190e-01,  8.22837338e-01,
 -1.03979123e-01,  1.19734567e+00, -2.86302983e-01,  1.19587413e+00,
 -1.04899709e+00, -1.02309750e+00,  3.12182463e-01,  8.13775741e-01,
 -2.07568463e+00,  1.89554291e+00, -8.57294256e-02, -1.51450872e+00,
 -3.17747457e-01, -2.32852824e+00,  1.80680637e+00, -2.68122580e+00,
 -2.64910937e+00, -1.82634665e+00,  2.84668775e+00,  2.63279705e+00,
 -7.97376823e-01, -2.40315313e+00, -2.32719158e+00, -2.06505708e+00,
 -1.94450865e+00,  7.51190165e-01,  4.74458611e-01,  2.95138700e+00,
 -1.48551164e+00, -1.10398012e+00,  1.47634656e+00,  4.85802572e-01,
  2.02189545e+00,  9.30388107e-01,  2.26386750e+00, -1.26556836e+00,
 -2.74761011e+00,  2.30778239e+00,  2.00000000e+00,  2.40867578e+02,
  2.00534759e-07,  4.58762689e+01,  3.87933152e+00,  6.25190669e+01,
  2.95285459e+01,  6.99065702e+01,  5.34357193e+01,  3.51416097e+01,
  6.47487196e+01,  4.51967776e+01,  1.97083558e+01,  7.33639649e+01,
  2.36844986e+01,  2.15655457e+01,  5.82320813e+01,  1.76205244e+01,
  3.15966249e+01,  1.60749761e+01,  9.47283674e+00,  1.54619310e+01,
  1.69676624e+01,  1.11528256e+01,  1.94317008e+01,  4.89277482e+01,
  5.05596999e+01,  6.56128880e+01,  5.51174489e+01,  4.06911313e+01,
  6.47083324e+01,  5.63114516e+01,  7.89748328e+01, -5.98751132e-01,
 -1.30303409e-01, -1.16801157e+00,  1.07733369e-01,  1.52647514e-01,
  2.24974452e-02,  2.11327971e-01,  8.94391439e-01, -2.69710024e-01,
  6.88924693e-01, -1.11581239e+00, -7.93942798e-02,  1.18863723e+00,
  5.49420702e-01, -4.96893502e-01,  3.26964941e-01,  3.09531403e-01,
  8.11048152e-02, -3.33819098e-01, -2.88135267e-01,  1.10661984e-01,
  1.90091368e-02, -2.51503808e-01,  4.92384731e-01, -3.79928027e-01,
  2.86689298e-01, -4.76944567e-01, -5.35379881e-01,  2.31465414e-01,
 -8.26538332e-01, -5.67122816e-01, -3.20447941e-01,  1.22628324e+00,
 -2.43889761e+00,  2.81854278e+00, -1.79457807e+00,  1.36679142e+00,
  3.30264996e-01,  3.10108540e+00,  2.64737958e+00, -1.57347227e+00,
 -2.21526177e+00,  2.59175820e+00,  2.25070199e+00,  1.12541186e+00,
 -2.82980309e+00, -1.42995085e+00,  5.33606070e-01, -1.38977104e+00,
 -1.04352140e+00, -1.29700559e+00,  2.27847851e+00, -2.90858055e+00,
  2.30078165e+00, -1.78821048e+00, -2.35026293e+00,  2.02609447e+00,
  1.58104989e+00, -7.95964198e-01,  4.93209665e-01,  2.00000000e+00,
  1.10419204e+01,  2.00534759e-07,  3.92067264e+01,  4.04211479e+01,
  7.34906530e+01,  6.81351478e+01,  6.98608295e+01,  3.50312181e+01,
  1.75293160e+01,  3.52938881e+01,  5.64160765e+01,  2.69432074e+01,
  2.78112488e+01,  5.33114669e+00,  6.81526425e+01,  5.25719633e+01,
  5.22515129e+01,  1.29905209e+01,  5.93202046e+01,  6.92934960e+01,
  3.30461992e+01,  1.90957531e+01,  1.25498465e+01,  2.41252642e+01,
  5.53308487e+01,  1.60268732e+01,  5.80744963e+01,  5.86480937e+01,
  3.91706858e+01,  3.18890594e+01,  4.87767051e+00,  4.08807876e+01,
  8.40731243e-01,  3.80820398e-01,  5.90980431e-01,  5.55681326e-01,
  4.23939687e-01, -6.97381578e-01,  5.53867118e-01,  3.03481070e-01,
  8.00292194e-01,  1.08661558e-01,  3.75925884e-01, -3.00108589e-01,
 -7.96454455e-01, -9.53839]
# ***************************************************************
# The corresponding costs are :
# Cost contribution 1 (Closest Distance to Sun): 0
# Cost contribution 2 (Distance away from Sun): 0.012342662655707196
# Cost contribution 3 (Distances from each other): -0.04484537035330449
# Total cost:  -0.0325027076975973
# -0.0325027076975973
# The minimum cost is:  [-0.03250271]
# Total run time: 17322.47993707657 seconds




# The best cost SO FAR is :  1.3638758262295907
# The best design variables SO FAR are:
# [ 2.00000000e+00,  9.05209701e+01,  2.00534759e-07,  7.13950863e+01,
#   6.31996066e+01,  3.31514634e+01,  8.53353191e+01,  4.28097922e+01,
#   6.16187172e+01,  1.96044585e+01,  4.21740409e+01,  2.17044208e+01,
#   5.16437927e+01,  6.60040107e+01,  2.43081612e+01,  5.50724937e+01,
#   2.92412792e+01,  2.71090587e+01,  2.31041847e+01,  4.77896158e+01,
#   6.55774971e+01,  7.94888027e+01,  2.06562789e+01,  7.35147495e+01,
#   1.68127541e+01,  2.73249669e+01,  3.67964743e+01,  6.34745088e+01,
#   9.35941524e+01,  2.71403236e+01,  7.30127995e+01,  9.17716919e+01,
#   4.40888060e+01,  3.87236210e+01,  2.27890394e+01,  1.03378479e+01,
#   1.01515060e+02,  7.56045843e+01,  2.27823976e+01,  4.07558360e+01,
#   5.98769646e+01,  6.72659936e+01,  1.52762060e+01,  5.46024839e+01,
#   2.12082385e+01,  5.12561864e+01,  7.19954786e+01,  3.03588364e+01,
#   5.85177855e+01,  3.13289481e+01,  2.24853743e+01,  3.49821757e+01,
#   5.71797344e+01,  3.91834182e-01, -3.31706317e-01, -1.79052044e-01,
#  -5.46127782e-01,  2.20578289e-01, -2.11878391e-01,  5.61967970e-01,
#   1.51711418e-01, -4.87722045e-01, -4.06048658e-01, -4.80379173e-01,
#   5.66490817e-01,  6.37967083e-01, -2.38765854e-01, -7.26088074e-01,
#  -2.66208889e-01,  9.37627915e-02,  3.66287687e-01,  6.22156871e-01,
#  -1.11059250e-01, -3.33780163e-01, -9.85894294e-02,  3.84987782e-01,
#  -3.07980748e-01, -2.93201963e-01, -1.73996100e-01, -7.46049079e-01,
#   1.97460546e-01,  2.37441683e-02,  1.37266196e-01, -1.94025467e-01,
#   8.64959956e-01,  5.23467670e-01,  4.10800404e-01, -4.20687923e-01,
#   6.37732303e-01,  5.24229723e-01, -2.18230243e-01, -1.31677758e-01,
#  -4.76886657e-02,  1.24922411e-01, -1.23144463e-01, -5.81975302e-02,
#  -3.11690460e-01,  3.72548807e-03,  5.35746191e-01, -5.52883975e-01,
#   8.98507649e-01, -2.23810570e-01, -2.23456576e-01, -1.36120068e+00,
#   2.17811462e+00,  1.98511512e+00, -3.87431776e-01, -8.77771940e-01,
#  -1.41989553e+00, -1.12648868e-01,  3.97157199e-01,  8.42509016e-03,
#  -1.45023233e+00,  1.15970575e+00,  3.16863704e-01, -8.96202463e-01,
#   2.14032303e+00, -8.42122450e-01,  8.08537590e-01, -1.54494524e+00,
#  -1.58362062e+00, -1.52677444e+00, -1.24898490e+00, -1.97013978e+00,
#   3.15030323e-01,  8.96974225e-01,  2.50727599e-01,  1.87321706e+00,
#   1.76541393e+00, -4.27172002e-01, -1.58027314e+00,  1.06403755e+00,
#   8.00677724e-01,  8.92691693e-01, -3.11444006e-01, -3.04681291e-01,
#  -1.26227906e+00,  2.54957786e-01, -7.88441018e-01, -8.30692773e-01,
#   2.08921869e+00, -1.28386787e+00,  3.36181978e-01, -2.96282887e-01,
#  -2.11221490e+00, -1.19518947e+00, -1.14965783e+00, -4.45662527e-01,
#   2.94400988e-01,  1.11354451e+00,  1.20404903e+00,  5.84440601e-01,
#  -2.12543057e+00]
# The corresponding cost breakdown is:
# 1.3638758262295907
# The best cost SO FAR happens in experiment with : NumSeg=50, SegLength=90, constant_angles=0, max_variation=0.0



# # The best cost SO FAR is :  1.3854162491251985
# # The best design variables SO FAR are:
# var_default = [ 1.00000000e+00,  3.46234337e+01,  2.00534759e-07,  5.64858776e+01,
#   5.76874028e+01,  4.50253965e+01,  9.56119443e+01,  6.03082354e+01,
#   6.60286773e+01,  1.48075649e+01,  7.71335207e+01,  3.86297419e+01,
#   7.88508649e+01,  3.83691627e+01,  5.50484320e+01,  1.84291581e+01,
#   6.76330347e+01,  9.15649343e+01,  1.39915900e+01,  9.58292833e+00,
#   1.16934409e+02,  3.88053033e+01,  6.41131259e+01,  3.67690652e+00,
#   8.50950005e+01,  2.42167848e+01,  4.48397744e+01,  5.09907607e+01,
#   4.52220919e+01,  3.23424956e+01,  7.85267853e+01,  2.54703661e+01,
#   9.77190471e+01,  5.87917222e+01,  1.04129610e+02,  3.22021231e+01,
#   1.75224077e+01,  7.73977879e+01,  2.11534422e+01,  1.10102346e+02,
#   6.28283044e+01,  5.93024588e+01,  8.40477437e+01,  5.89612231e+01,
#   7.58320669e+01,  7.08897565e+01,  7.99202818e+01,  8.51077205e+01,
#   7.39907078e+00,  8.82672887e+01,  8.70307119e+01,  8.31232645e+01,
#   7.25653138e+01,  7.38771582e+00,  1.30892269e+01,  1.22383501e+01,
#   3.76311383e+01,  9.76042914e+00,  5.68726324e+01,  4.52164168e+01,
#   2.20526581e+01,  8.47404185e+01,  3.75071570e+01,  3.54708000e-01,
#   2.03804647e-01,  1.09770410e+00,  1.42532690e-01,  2.53333291e-01,
#   2.64732836e-03,  2.92148533e-01, -5.87108958e-01,  9.95839691e-02,
#   8.36184606e-01, -8.30868375e-01,  1.18012765e+00,  7.17307955e-01,
#   1.15919458e+00,  9.60521780e-01,  4.77781883e-01, -8.33028683e-01,
#   2.39136280e-01,  8.09981808e-01,  5.37067532e-01, -6.96838040e-01,
#  -3.89529805e-01,  1.06124620e+00,  2.84933054e-01,  4.89962266e-01,
#  -7.84925703e-01,  1.08025772e+00,  1.11048262e-01,  4.64425166e-01,
#  -7.54147771e-01, -6.80889630e-02, -6.77793700e-01, -8.54397521e-01,
#   8.86234080e-01, -3.07627312e-01, -4.65023207e-01, -1.20213709e-01,
#   1.34625224e+00,  4.40295307e-01, -4.67197608e-02,  4.57113716e-01,
#   5.03187898e-02, -7.70409708e-01, -4.62061115e-01, -1.11437606e+00,
#   6.21318342e-01, -1.94679906e-01,  1.17459483e+00,  9.96555062e-01,
#  -5.07884908e-01,  3.89510951e-01, -7.92739851e-01,  4.21093883e-01,
#  -8.09102802e-01, -4.46910072e-01, -3.07770138e-01,  9.26425637e-01,
#   9.96717821e-01,  1.23934198e-01,  6.29104231e-01,  2.97515922e+00,
#  -2.22483897e+00, -1.50098622e+00,  8.51412419e-02, -2.81614028e+00,
#  -2.04618044e-01,  1.37496257e+00, -6.87050595e-01,  1.04600787e+00,
#   2.17889960e-01,  1.31508085e+00,  3.44130529e-01,  1.70016473e+00,
#   2.79720550e-01,  9.00173588e-01,  1.36514852e+00, -2.97240080e+00,
#  -6.18225269e-01, -3.04380403e+00,  1.96935102e+00, -4.62763766e-01,
#  -1.86181513e-01,  4.85231928e-01,  2.76800670e+00, -9.91607246e-01,
#   1.65137719e+00, -3.02646448e+00,  4.06372769e-02, -5.58231821e-01,
#  -1.47388588e+00, -9.23667127e-01,  3.05872655e+00, -8.14843796e-01,
#   1.24547961e-01,  1.23401404e+00,  1.35976100e-01, -2.08260695e+00,
#  -1.23573709e+00,  5.54336135e-01,  2.11002438e+00,  1.17352003e+00,
#  -2.46702809e+00,  1.73460521e+00,  3.09292047e-01,  2.70882367e+00,
#  -9.11644269e-01, -1.89918789e+00,  2.90306552e+00, -1.79524235e-01,
#   1.51703021e+00, -1.87087727e+00, -1.54904578e+00,  1.77800704e+00,
#   2.41410950e+00, -9.13847842e-01,  2.37426527e+00, -1.53131810e+00,
#   2.24286035e+00,  5.23802413e-01,  7.35798182e-01]

# # The corresponding cost breakdown is:
# # 1.3854162491251985
# # The best cost SO FAR happens in experiment with : NumSeg=30, SegLength=50, constant_angles=0, max_variation=0.0



if args.inputfilename == None:
    var = var_default
else:
    var = np.loadtxt(args.inputfilename)

NumSeg = round((np.array(var).shape[0]-3)/3)

AU = 1.496e11 # Astronomical Unit (m)

TOLNEO=100000*1000/AU
TOLEarth=0.1

desHoverTime = 60 # Desired hover time (days) 
constant_angles = 0 # constant vs smooth angles
# solarSysExit = 1

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

print("Earth velocities:", Earth.velocities.shape)
print("Earth positions:", Earth.positions.shape)
# print("NEO velocities:", NEO.velocities.shape)
# print("NEO positions:", NEO.positions.shape)

if solarSysExit != 1:
    print("Running simulation with one sail:")
    bodies = {Earth, targetObject}

    #Desired designs and weights
    w = [0] * 6
    w[0] = 1 # Total energy
    w[1] = 1 # Hover Time
    w[2] = 10 # Hover Time 2
    w[3] = 1 # Return
    w[4] = 1 # Closest Distance to Sun
    w[5] = 10 # Approach velocity to NEO

    solver = lightSailSolver(var,bodies)

    print("***************************************************************")
    print("The design variables are:")
    np.set_printoptions(threshold=1000000000)
    print(np.array2string(var, separator=', '))
    print("***************************************************************")
    print("desHoverTime:", desHoverTime)
    print("constant_angles:", constant_angles)
    print("T:", T)
    print("TOLNEO:", TOLNEO)
    print("TOLEarth:", TOLEarth)
    print("savetraj:", savetraj)
    print("NumSeg:", NumSeg)
    print("dT:", dT)
    if args.outputfilename == None:
        solver.runSim(desHoverTime, constant_angles, T, TOLNEO, TOLEarth, savetraj, NumSeg, dT, useEndCond = useEndCond, NEOname = NEOname)
    else:
        solver.runSim(desHoverTime, constant_angles, T, TOLNEO, TOLEarth, savetraj, NumSeg, dT, traj_output_filename=args.outputfilename, useEndCond = useEndCond, NEOname = NEOname)

    if trackNEO == 1:
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

    print("NEOPos", NEOPos)

    if trackNEO == 1:
        # Find the index of the smallest distance
        min_distance_index = np.argmin(distances)

        minDist=np.min(distances)

        # Get the corresponding time step
        min_distance_time = ToF[min_distance_index]

else:

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

    numSails = 30
    # NumSeg = round(((np.array(var).shape[0]-3)/3)/numSails)
    NumSeg = 30
    dv=3+NumSeg*3

    bodies = {Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune}

    #Desired designs and weights
    w = [0] * 3
    w[0] = 1 # Total energy
    w[1] = 1 # Hover Time
    w[2] = 10 # Hover Time 2

    solver = lightSailSolver(var[0:dv], bodies)
    solver.runSim(desHoverTime, constant_angles, T, TOLNEO, TOLEarth, savetraj, NumSeg, dT, NEOname = NEOname)
    dataSize = len(solver.simTime)

    mercuryPos = solver.mercuryPos 
    venusPos = solver.venusPos
    marsPos = solver.marsPos
    jupiterPos = solver.jupiterPos
    saturnPos = solver.saturnPos
    uranusPos = solver.uranusPos
    neptunePos = solver.neptunePos
    earthPos = solver.earthPos
    sunPos = solver.sunPos
    simTime = solver.simTime

    planet_data = [mercuryPos, venusPos, earthPos, marsPos, jupiterPos, saturnPos, uranusPos, neptunePos]
    light_sail_data = np.empty((numSails, dataSize, 3))
    sail_normal = np.empty((numSails, dataSize, 3))
    sail_flight_data= np.empty((numSails, len(solver.sailFlightPos), 3))

    for i in range(numSails):
        print("I am here..")
        solver = lightSailSolver(np.concatenate(([var[i*dv], var[1]], var[i*dv + 2:(i+1)*dv])), bodies)
        solver.runSim(desHoverTime, constant_angles, T, TOLNEO, TOLEarth, savetraj, NumSeg, dT)
        
        light_sail_data[i, :, :] = solver.sailPos
        sail_normal[i, :, :] = solver.sailNormal
        sail_flight_data[i,:,:] = solver.sailFlightPos

    print('I am done running the simulations.')
    export_to_tecplot_SolarExit('solarEscape.dat', simTime, sunPos, light_sail_data, sail_normal, sail_flight_data, planet_data, numSails)





# print(f"Shortest time to NEO: {min_distance_time} days")

# cost=w1*(min_distance_time)+w2*(minDist)
# print(cost)


###############################################################################################
#Plots
if plots==1 and solarSysExit != 1:
    # Plot alphaG vs ToF
    plt.figure(figsize=(10, 6))
    # plt.plot(ToF, np.mod(alphaG, 6.283))
    plt.plot(ToF, alphaG)
    # plt.plot(ToF, np.mod(alphaG, 6.283))
    plt.plot(ToF, alphaG)
    plt.xlabel('Time of Flight (days)')
    plt.ylabel('Cone Angle (rad)')
    plt.title('Cone Angle Variation Over Time')
    plt.grid(True)
    plt.show()
    # print("The alphas are: ", alphaG)
    # print("The gammas are: ", gammaG)
    # Plot gammaG vs ToF
    plt.figure(figsize=(10, 6))
    # plt.plot(ToF, np.mod(gammaG, 6.283))
    plt.plot(ToF, gammaG)
    # plt.plot(ToF, np.mod(gammaG, 6.283))
    plt.plot(ToF, gammaG)
    plt.xlabel('Time of Flight (days)')
    plt.ylabel('Clock Angle (rad)')
    plt.title('Clock Angle Variation Over Time')
    plt.grid(True)
    plt.show()

    if trackNEO == 1:
        # Plot distances vs ToF
        plt.figure(figsize=(10, 6))
        plt.plot(ToF, distances)
        plt.xlabel('Time of flight (days)')
        plt.ylabel('Distance (AU)')
        plt.title('Distance Between Light sail and NEO Over Time')
        plt.grid(True)
        plt.show()

#Plotting sunPos, earthPos, NEOPos, sailPos, and simTime in a movie:
if movie==1 and solarSysExit != 1:
    # Set up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    # To view from above:
    ax.elev = 90
    ax.azim = 0

    # Initialize empty lines for each position
    sun_line, = ax.plot([], [], [], 'yo', markersize=10, label='Sun')
    earth_line, = ax.plot([], [], [], 'bo', markersize=5, label='Earth')
    if trackNEO == 1:
        neo_line, = ax.plot([], [], [], 'go', markersize=5, label='NEO')
    sail_line, = ax.plot([], [], [], 'ro', markersize=3, label='Sail')

    # Update function for animation
    def update(frame):
        ax.clear()  # Clear the axes

        # Set labels and title
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        ax.set_title('Optimum Light Sail Trajectory to NEO 2016 VO1')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])

        # Plot sun
        sun_line = ax.plot([sunPos[frame][0]], [sunPos[frame][1]], [sunPos[frame][2]], 'yo', markersize=10, label='Sun')[0]

        # Plot Earth
        earth_line = ax.plot([earthPos[frame][0]], [earthPos[frame][1]], [earthPos[frame][2]], 'bo', markersize=3.5, label='Earth')[0]

        # Plot NEO
        if trackNEO == 1:
            neo_line = ax.plot([NEOPos[frame][0]], [NEOPos[frame][1]], [NEOPos[frame][2]], 'go', markersize=2, label='NEO')[0]

        # Plot Sail
        sail_line = ax.plot([sailPos[frame][0]], [sailPos[frame][1]], [sailPos[frame][2]], 'ro', markersize=1, label='Sail')[0]

        # # Update simulation time text
        # time_text.set_text(f"Simulation Time: {simTime[frame]:.2f} days")
        # Add text for simulation time
        time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
        time_text.set_text(f"Simulation Time: {simTime[frame]:.2f} days")

        # Show legend
        ax.legend()

        return sun_line, earth_line, neo_line, sail_line, time_text

    # Add text for simulation time
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # Set title
    ax.set_title('Optimum Light Sail Trajectory to NEO 2016 VO1')

    # Show legend
    ax.legend()

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(simTime), interval=0.4, blit=False)

    # Save the animation
    ani.save('example.gif', writer='pillow', fps=12)

    # Show plot
    plt.show()

# Record the end time
end_time = time.time()

# Calculate the total run time
total_run_time = end_time - start_time

# Print the total run time
print(f"Total run time: {total_run_time} seconds")


