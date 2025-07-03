import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.interpolate import lagrange
import argparse
import pandas as pd

def export_to_tecplot(filename, time_steps, sun_data, light_sail_data, earth_data, neo_data, sail_normal,sailFlightPos, trackNEO=1):
    with open(filename, 'w') as f:
        # Write title and variable definitions
        f.write('TITLE = "Light Sail Trajectory"\n')
        f.write('VARIABLES = "X", "Y", "Z", "NX", "NY", "NZ"\n')
        # f.write('VARIABLES = "X", "Y", "Z"\n')
        
        # Loop through time steps and write zones for each entity
        for i, time in enumerate(time_steps):
            # Write zone for Sun
            f.write(f'ZONE T="Sun", SOLUTIONTIME={time}\n')
            f.write(f'{sun_data[i,0]} {sun_data[i,1]} {sun_data[i,2]} 0 0 0\n')
                
            # Write zone for Light Sail
            f.write(f'ZONE T="Light Sail", SOLUTIONTIME={time}\n')  
            f.write(f'{light_sail_data[i,0]} {light_sail_data[i,1]} {light_sail_data[i,2]} {sail_normal[i,0]} {sail_normal[i,1]} {sail_normal[i,2]}\n')
            
            # Write zone for Earth
            f.write(f'ZONE T="Earth", SOLUTIONTIME={time}\n')
            f.write(f'{earth_data[i,0]} {earth_data[i,1]} {earth_data[i,2]} 0 0 0\n')
                
            # Write zone for NEO
            if trackNEO == 1:
                f.write(f'ZONE T="NEO", SOLUTIONTIME={time}\n')
                f.write(f'{neo_data[i,0]} {neo_data[i,1]} {neo_data[i,2]} 0 0 0\n')
        
        f.write(f'ZONE T="Light Sail Trajectory"\n')
        for i in range(len(sailFlightPos)):
            f.write(f'{sailFlightPos[i,0]} {sailFlightPos[i,1]} {sailFlightPos[i,2]} 0 0 0\n')

        if trackNEO == 1:
            f.write(f'ZONE T="NEO Trajectory"\n')
            for i in range(len(sailFlightPos)):
                f.write(f'{neo_data[i,0]} {neo_data[i,1]} {neo_data[i,2]} 0 0 0\n')
        
        f.write(f'ZONE T="Earth Trajectory"\n')
        for i in range(len(earth_data)):
            f.write(f'{earth_data[i,0]} {earth_data[i,1]} {earth_data[i,2]} 0 0 0\n')

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

class lightSailSolver:

    def __init__(self,var,bodies):
        self.var = var
        self.bodies = bodies

    def runSim(self, desHoverTime, constant_angles, T, TOL,TOL2, MakeMovie, NumSeg, dT, traj_output_filename = 'sail_trajectory', trackNEO = 1, useEndCond = 1, NEOname = 'Vesta'):
        '''
        deHoverTime : desired hover time (days)
        constant_angles:  = 1 if you need to make the angles change in a step-function fashion (or 0 for interpolation technique for smooth angles) 
        T : Maximum time
        TOL: Tolerance for reaching desired position (AU)
        TOL2: Tolerance for coming back to Earth (AU)
        MakeMovie: Create .dat file to send to Tecplot and export .csv file if MakeMovie = 1
        NumSeg: Number of time segments in trajectory plan
        dT: Time step (Days)
        traj_output_filename: if MakeMovie=1, export to [traj_output_filename].dat and [traj_output_filename].csv
        trackNEO: if trackNEO=1, track the NEO, if trackNEO=0, don't track NEO
        useEndCond: stop simulation when end condition is reached if useEndCond = 1
        '''

        var = self.var
        bodies = self.bodies
        self.desHoverTime = desHoverTime
        dT = float(dT)

        Earth = find_body(bodies, 'Earth')
        NEO = find_body(bodies, NEOname)
        Mercury = find_body(bodies, 'Mercury')
        Venus = find_body(bodies, 'Venus')
        Mars = find_body(bodies, 'Mars')
        Jupiter = find_body(bodies, 'Jupiter')
        Saturn = find_body(bodies, 'Saturn')
        Uranus = find_body(bodies, 'Uranus')
        Neptune = find_body(bodies, 'Neptune')
        Pluto = find_body(bodies, 'Pluto')

        # print("NEOname", NEOname)
        # print("NEO", NEO)

        M_NEO = NEO.mass if NEO is not None else 0
        M_earth = Earth.mass
        M_Mercury = Mercury.mass if Mercury is not None else 0
        M_Venus = Venus.mass if Venus is not None else 0
        M_Mars = Mars.mass if Mars is not None else 0
        M_Jupiter = Jupiter.mass if Jupiter is not None else 0
        M_Saturn = Saturn.mass if Saturn is not None else 0
        M_Uranus = Uranus.mass if Uranus is not None else 0 
        M_Neptune = Neptune.mass if Neptune is not None else 0 
        M_Pluto = Pluto.mass if Pluto is not None else 0 


        earthPos = Earth.positions
        earthVel = Earth.velocities
     
        num_time_steps = round(T / dT) + 3

        NEOPos = NEO.positions if NEO is not None else np.zeros((num_time_steps, 3))
        NEOVel = NEO.velocities if NEO is not None else np.zeros((num_time_steps, 3))
        mercuryPos = Mercury.positions if Mercury is not None else np.zeros((num_time_steps, 3))
        mercuryVel = Mercury.velocities if Mercury is not None else np.zeros((num_time_steps, 3))
        venusPos = Venus.positions if Venus is not None else np.zeros((num_time_steps, 3))
        venusVel = Venus.velocities if Venus is not None else np.zeros((num_time_steps, 3))
        marsPos = Mars.positions if Mars is not None else np.zeros((num_time_steps, 3))
        marsVel = Mars.velocities if Mars is not None else np.zeros((num_time_steps, 3))
        jupiterPos = Jupiter.positions if Jupiter is not None else np.zeros((num_time_steps, 3))
        jupiterVel = Jupiter.velocities if Jupiter is not None else np.zeros((num_time_steps, 3))
        saturnPos = Saturn.positions if Saturn is not None else np.zeros((num_time_steps, 3))
        saturnVel = Saturn.velocities if Saturn is not None else np.zeros((num_time_steps, 3))
        uranusPos = Uranus.positions if Uranus is not None else np.zeros((num_time_steps, 3))
        uranusVel = Uranus.velocities if Uranus is not None else np.zeros((num_time_steps, 3))
        neptunePos = Neptune.positions if Neptune is not None else np.zeros((num_time_steps, 3))
        neptuneVel = Neptune.velocities if Neptune is not None else np.zeros((num_time_steps, 3))
        AU = 1.496e11 # Astronomical Unit (m)
        dt = dT*24*60*60 # Converting dT to seconds
        degree = int(var[0]) # Degree of polynomial interpolation
        initial_launch = var[1] # initial launch time (in days after simulation starts)
        vInit = var[2] # Magnitude of initial launch velocity
        distNEO = 10 # Initial hypothetical distance (AU)
        reachedNEO = 0 # Initially 0, set to 1 once sail has reached within TOL of NEO
        endCond = 0 # End condition is initially 0, set to 1 once sail has 1.) reached within TOL of NEO and 2.) returned to within 100000km of Earth
        t=0 # total time (s)
        t_angle=0 # Angles start time (after launch) (s)
        t_days=0 # total time (days)
        # setInitDir = 0 if var[24]<0.5 else 1 # which side of earth does the sail begin at at initial_launch (1: begins on side of Earth in the direction Earth is moving towards, 0: begins on side of Earth away from the direction Earth is moving)
        setInitDir = 1

        # Constants
        G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
        M_sun = 1.989e30  # Mass of the Sun (kg)
        M_sail = 14.6*10**(-3) # Mass of light sail (kg)
        Radius_Earth=6378000/AU # Earth's radius (AU)
        # AtmosphereDist=10000000/AU # Distance from Earth's surface to atmosphere (AU)
        SlingshotDist = 1000000*1000/AU # Distance from Earth's surface to location after slingshotting away from Earth's gravity (1,000,000km converted to AU)
        rInit=Radius_Earth+SlingshotDist # Initial distance of Sail from Earth's center (AU)
        # M_NEO = UNKNOWN # Mass of the NEO (kg)
        beta=0.16 # Lightness number (dimensionless) - see Mclnnes eq. 2.25a p. 40 and BLISS paper
        muEarth = G * M_earth/(AU**3)
        muMercury = G * M_Mercury/(AU**3)
        muVenus = G * M_Venus/(AU**3)
        muMars = G * M_Mars/(AU**3)
        muJupiter = G * M_Jupiter/(AU**3)
        muSaturn = G * M_Saturn/(AU**3)
        muUranus = G * M_Uranus/(AU**3)
        muNeptune = G * M_Neptune/(AU**3)
        muSun=G*(M_sun)/(AU**3)
        # muNEO=G*M_NEO*M_sail
    
        # Create arrays to store positions at each time step
        p = np.array([0,0,1]) # Normal vector to Earth's orbit
        sailPos = np.empty((0, 3))
        sunPos = np.empty((0, 3))
        sailNormal = np.empty((0,3)) 
        sailFlightPos = np.empty((0,3))
        sailVelocities = []
        sailSunDistArray = []
        NEO_sail_Vel=[]
        distances = [] #distances between sail and NEO over the full simulation time (after launch)
        distancesToNEOFullTime = [] #distances between sail and NEO over the full simulation time (including pre-launch)
        simTime=[]
        ToF=[]
        alphaG=[]
        gammaG=[]

        sailActive=[] #track status: 0=not in flight, 1=in flight/active

        time_segments = calc_time_segments(var[3:NumSeg+3])
        cone_angle_var = var[NumSeg+3:2*NumSeg+3]
        clock_angle_var = var[2*NumSeg+3:3*NumSeg+3]
        segmentsClocks, segmentsCones = create_angle_functions(time_segments, clock_angle_var, cone_angle_var, degree)

        # print("Time segments:", time_segments, "\n Cone angles: ", cone_angle_var, "\n Clock angles:", clock_angle_var, "\n\n")

        # Main loop
        while(t_days < T and endCond == 0): 
            # Simulation time
            simTime=np.append(simTime, t_days)
            
            # Calculate Cartesian coordinates of the Sun
            sunPos=np.vstack((sunPos,[0, 0, 0]))
            
            # Start loop for sail to go after NEO
            if (t_days < initial_launch):
                # Calculate and set sail position such that it is 
                # 1.) rInit from Earth's center, 
                # 2.) In the same orbital plane as Earth, 
                # 3.) Along the line perpendicular to the Earth's sun-line 
                # (two possible such positions exist):
                zS=0
                xS,yS=initialPos(earthPos[len(simTime)-1,0],earthPos[len(simTime)-1,1],rInit)
                sailPos=np.vstack((sailPos,[xS,yS,zS]))
                sailNormal=np.vstack((sailNormal,[0,0,0]))
                r=np.array([xS,yS,zS])

                #Determine which side of earth the solar sail is on (if 1, is on the side the earth is moving towards, if 0, is on the side the Earth is moving away from):
                if t == 0 :
                    dirIndicator = 1 if np.dot(earthVel[len(simTime)],np.squeeze(sailPos[len(simTime)-1]-earthPos[len(simTime)-1]))>0 else 0 
                else:
                    dirIndicator = 1 if np.dot(earthVel[len(simTime)-1],np.squeeze(sailPos[len(simTime)-1]-earthPos[len(simTime)-1]))>0 else 0 
               
                # Set the sail to the correct side of Earth based on parameter setInitDir
                if setInitDir != dirIndicator:
                    sailPos[-1,:] = 2*earthPos[len(simTime)-1,:]-sailPos[-1,:]
                    r = sailPos[-1,:]
                
                # Set sail initial position approximately tangent to Earth's orbit 
                r0 = r-earthPos[len(simTime)-1,:]
                r0 = r0/np.linalg.norm(r0)
                v = vInit*r0
                sailActive.append(0)

            # Sail Launched
            else:
                # Define cone and clock angles
                if constant_angles == 1:
                    alpha, gamma = parse_angles(time_segments,cone_angle_var,clock_angle_var,t_days)
                else:
                    alpha = find_value_at_time(segmentsCones, time_segments, t_days)
                    gamma = find_value_at_time(segmentsClocks, time_segments, t_days)
                # print("Time:", t_days, ", Cone angle: ", alpha, ", Clock angle:", gamma, "\n\n")

                # Ensure angles are within allowable range
                alpha = np.clip(alpha, -np.pi / 2, np.pi / 2)
                gamma = np.clip(gamma, -np.pi, np.pi)

                if t_days == initial_launch:
                    p=np.array([0,0,1]) # Normal vector to Earth's orbit
                else:
                    p = np.cross(r,v)

                # Calculate sail normal
                n1 = np.cos(alpha) * (r/np.linalg.norm(r)) # Radial
                n2 = np.cos(gamma) * np.sin(alpha) * (p/np.linalg.norm(p)) # Orbital normal
                n3 = np.sin(alpha) * np.sin(gamma) * np.cross(p/np.linalg.norm(p),r/np.linalg.norm(r)) # Transverse

                n = n1 + n2 + n3

                sailNormal = np.vstack((sailNormal,n))

                # Forward Euler velocity solver
                if np.cos(alpha) <= 0:  # sail perpendicular to or away from the sun -> no solar force
                    # print("dir:", np.dot(r, n)) #check on force direction, positive dot product between n and r indicates force away from sun, negative is force towards sun
                    v = v - dt * muSun * (r / np.linalg.norm(r)**3) \
                        - dt * muEarth * (earthPos[len(simTime)-1, :] - r) / np.linalg.norm(earthPos[len(simTime)-1, :] - r)**3 \
                        - dt * muMercury * (mercuryPos[len(simTime)-1, :] - r) / np.linalg.norm(mercuryPos[len(simTime)-1, :] - r)**3 \
                        - dt * muVenus * (venusPos[len(simTime)-1, :] - r) / np.linalg.norm(venusPos[len(simTime)-1, :] - r)**3 \
                        - dt * muMars * (marsPos[len(simTime)-1, :] - r) / np.linalg.norm(marsPos[len(simTime)-1, :] - r)**3 \
                        - dt * muJupiter * (jupiterPos[len(simTime)-1, :] - r) / np.linalg.norm(jupiterPos[len(simTime)-1, :] - r)**3 \
                        - dt * muSaturn * (saturnPos[len(simTime)-1, :] - r) / np.linalg.norm(saturnPos[len(simTime)-1, :] - r)**3 \
                        - dt * muUranus * (uranusPos[len(simTime)-1, :] - r) / np.linalg.norm(uranusPos[len(simTime)-1, :] - r)**3 \
                        - dt * muNeptune * (neptunePos[len(simTime)-1, :] - r) / np.linalg.norm(neptunePos[len(simTime)-1, :] - r)**3
                else:
                    # print("dir:", np.dot(r, n)) #check on force direction, positive dot product between n and r indicates force away from sun, negative is force towards sun
                    v = v + dt * beta * (muSun / np.linalg.norm(r)**2) * (np.cos(alpha))**2 * (n / np.linalg.norm(n)) \
                        - dt * muSun * (r / np.linalg.norm(r)**3) \
                        - dt * muEarth * (earthPos[len(simTime)-1, :] - r) / np.linalg.norm(earthPos[len(simTime)-1, :] - r)**3 \
                        - dt * muMercury * (mercuryPos[len(simTime)-1, :] - r) / np.linalg.norm(mercuryPos[len(simTime)-1, :] - r)**3 \
                        - dt * muVenus * (venusPos[len(simTime)-1, :] - r) / np.linalg.norm(venusPos[len(simTime)-1, :] - r)**3 \
                        - dt * muMars * (marsPos[len(simTime)-1, :] - r) / np.linalg.norm(marsPos[len(simTime)-1, :] - r)**3 \
                        - dt * muJupiter * (jupiterPos[len(simTime)-1, :] - r) / np.linalg.norm(jupiterPos[len(simTime)-1, :] - r)**3 \
                        - dt * muSaturn * (saturnPos[len(simTime)-1, :] - r) / np.linalg.norm(saturnPos[len(simTime)-1, :] - r)**3 \
                        - dt * muUranus * (uranusPos[len(simTime)-1, :] - r) / np.linalg.norm(uranusPos[len(simTime)-1, :] - r)**3 \
                        - dt * muNeptune * (neptunePos[len(simTime)-1, :] - r) / np.linalg.norm(neptunePos[len(simTime)-1, :] - r)**3

                r = r + dt * v

                sailVel = v
                sailPos = np.vstack((sailPos,r))
                sailFlightPos = np.vstack((sailFlightPos,r))
                sailVelocities.append(v)  # Store velocity for energy calculation
                
                #Calculate distance between sail and sun
                sailSunDist = np.linalg.norm(r)
                sailSunDistArray=np.append(sailSunDistArray,sailSunDist)

                # Store distances between sail and NEO
                if trackNEO == 1:
                    distNEO=np.sqrt((r[0] - NEOPos[len(simTime)-1,0])**2 + (r[1] - NEOPos[len(simTime)-1,1])**2 + (r[2] - NEOPos[len(simTime)-1,2])**2)
                    distances=np.append(distances,distNEO)     


                t_angle=t_angle+dT
                ToF=np.append(ToF,t_angle)
                alphaG=np.append(alphaG,alpha)
                gammaG=np.append(gammaG,gamma)
                sailActive.append(1)
                

            t=t+dt # time progression in seconds
            t_days=dT+t_days

            if trackNEO == 1:
                distNEO=np.sqrt((r[0] - NEOPos[len(simTime)-1,0])**2 + (r[1] - NEOPos[len(simTime)-1,1])**2 + (r[2] - NEOPos[len(simTime)-1,2])**2)
                distancesToNEOFullTime.append(distNEO)

            if trackNEO == 1 and useEndCond == 1:
                # Determine simulation end condition
                if reachedNEO == 0:
                    if distNEO < TOL:
                        reachedNEO = 1
                else:
                    # Check distance between Earth and sail
                    distEarth=np.sqrt((r[0] - earthPos[len(simTime)-1,0])**2 + (r[1] - earthPos[len(simTime)-1,1])**2 + (r[2] - earthPos[len(simTime)-1,2])**2)

                    # Check relative velocity magnitude between Earth and Sail (km/s)
                    earth_sail_Vel = np.linalg.norm(earthVel[len(simTime)-2,:]-sailVel)*(AU/1000)


                    if distEarth < TOL2:
                        endCond = 1
        
        self.sailSunDistArray=sailSunDistArray
        self.sailPos = sailPos

        self.mercuryPos = mercuryPos
        self.venusPos = venusPos
        self.marsPos = marsPos
        self.jupiterPos = jupiterPos
        self.saturnPos = saturnPos
        self.uranusPos = uranusPos
        self.neptunePos = neptunePos


        self.earthPos = earthPos
        self.earthVel = earthVel
        self.NEOPos = NEOPos
        self.NEOVel=NEOVel
        self.sunPos = sunPos
        self.sailNormal = sailNormal
        self.sailFlightPos=sailFlightPos
        self.distances = distances
        self.simTime=simTime
        self.t_angle=t_angle
        self.ToF=ToF
        self.alphaG=alphaG
        self.gammaG=gammaG
        self.endCond = endCond
        self.t_days = t_days
        self.sailVelocities = sailVelocities
        distEarth=np.sqrt((r[0] - earthPos[len(simTime)-1,0])**2 + (r[1] - earthPos[len(simTime)-1,1])**2 + (r[2] - earthPos[len(simTime)-1,2])**2)
        earth_sail_Vel = np.linalg.norm(earthVel[len(simTime)-2,:]-sailVel)*(AU/1000)
        self.earth_sail_Vel=earth_sail_Vel
        self.finalDistEarth = distEarth
        self.reachedNEO = reachedNEO
        self.sailActive = sailActive
        self.M_sail = M_sail
        self.dT = dT
        self.initial_launch = initial_launch
        self.distancesToNEOFullTime = distancesToNEOFullTime
        
        # Create .dat file to send to Tecplot, also create and save dataframe for easier access to data
        if MakeMovie==1:
            export_to_tecplot(traj_output_filename+".dat", self.simTime, self.sunPos, self.sailPos, self.earthPos, self.NEOPos, self.sailNormal, self.sailFlightPos)
            print("saving")
            # print(f"Lengths - sailPos: {len(sailPos)}, earthPos: {len(earthPos)}, NEOPos: {len(NEOPos)}, time_array: {len(simTime)}")

            # Ensure all arrays have the same length
            min_length = min(len(self.simTime), len(self.sailActive), len(self.sailPos), len(self.earthPos))

            # Truncate arrays to the same length
            self.simTime = self.simTime[:min_length]
            self.sailActive = self.sailActive[:min_length]
            self.sailPos = self.sailPos[:min_length]
            self.earthPos = self.earthPos[:min_length]

            df = pd.DataFrame({
                "Time": self.simTime,
                "sailActive": self.sailActive,
                "sailPosX": self.sailPos[:,0],
                "sailPosY": self.sailPos[:,1],
                "sailPosZ": self.sailPos[:,2],
                "earthPosX": self.earthPos[:,0],
                "earthPosY": self.earthPos[:,1],
                "earthPosZ": self.earthPos[:,2],
                "distanceToNEO": np.array(self.distancesToNEOFullTime)
            })

            df.to_csv(traj_output_filename+".csv")
            
            

                
    
    def calcCost(self, w, TOL, TOL1, printStatements = 1):
        Penalty=100000
        distances = self.distances
        ToF = self.ToF
        t_angle=self.t_angle
        endCond = self.endCond
        t_days = self.t_days
        sailSunDistArray = self.sailSunDistArray
        finalDistEarth = self.finalDistEarth
        reachedNEO = self.reachedNEO
        earth_sail_Vel=self.earth_sail_Vel
        sailVelocities = self.sailVelocities
        simTime=self.simTime
        dT = self.dT
        desHoverTime = 60/dT
        distNEO_times_within_tolerance = 0
        total_energy = 0
        M_sail=self.M_sail
        NEOVel = self.NEOVel
        AU = 1.496e11 # Astronomical Unit (m)

        # Closest distance to the sun
        sunClose = np.min(sailSunDistArray)

        # Total number of time steps light sail is within 100,000km or TOL of NEO 
        hoverTime = np.sum(np.array(distances) < TOL)

        # Find the index of the smallest distance
        min_distance_index = np.argmin(distances)

        # Total number of time steps light sail is within 0.1 AU of NEO 
        hoverTime2 = np.sum(np.array(distances[:min_distance_index]) < 0.1)
        
        # Find smallest distance, corresponding approach velocity, and time step
        minDist=np.min(distances) 

        NEOapproachVelCost = 0 if reachedNEO == 1 else np.linalg.norm(NEOVel[min_distance_index] - sailVelocities[min_distance_index]) * (AU/1000) 

        # Close distance between NEO and sail
        approxDist = 1 if minDist * Penalty < 500 else 0

        # Distance norm to NEO
        distNorm = np.sum(np.array(distances))/(len(distances))

        # Get the corresponding time step
        min_distance_time = ToF[min_distance_index]

        # Quantify cost for NEO approach and hover
        hoverCost = abs(desHoverTime-hoverTime)/desHoverTime if reachedNEO == 1 else minDist * Penalty
        hoverCost2 = abs(desHoverTime*2-hoverTime2)/(2*desHoverTime) if minDist < 0.1 else minDist * Penalty

        # Quantify cost for Earth return
        returnCost = 0 if endCond == 1 else finalDistEarth

        # Quantify cost for Earth approach velocity
        returnVelCost = earth_sail_Vel if endCond==1 else 0

        # Quantify cost for too close to sun
        sunCost = 0 if sunClose > 0.25 else Penalty

        # Quantify cost for total energy expenditure
        total_energy_to_NEO = np.sum([0.5 * M_sail * np.linalg.norm(v)**2 for v in sailVelocities[:min_distance_index]])
        total_energy = np.sum([0.5 * M_sail * np.linalg.norm(v)**2 for v in sailVelocities])
        total_energy_to_earth = np.sum([0.5 * M_sail * np.linalg.norm(v)**2 for v in sailVelocities[min_distance_index:]])
        energyCost = total_energy_to_earth*10**13 if approxDist == 1 or reachedNEO == 1 else total_energy_to_NEO*Penalty*10**11

        # Calculate cost
        cost = (
                w[0] * energyCost +
                w[1] * hoverCost + 
                w[2] * hoverCost2 +
                w[3] * returnCost +
                w[4] * sunCost + 
                w[5] * NEOapproachVelCost
              )
        
        if printStatements == 1:
            print(f"Time to minimum distance with NEO after launch: {min_distance_time} days")
            print(f"Time to minimum distance with NEO after start of simulation: {self.initial_launch + min_distance_time} days")
            print(f"Closest distance to NEO: {minDist} AU")
            print(f"Total flight time: {simTime[-1]} days")
            print("Returned to earth:", endCond)
        
            print("Cost contribution 1 (Total energy):", w[0] * energyCost)
            print("Cost contribution 2 (Hover Time):", w[1] * hoverCost)
            print("Cost contribution 3 (Hover Time 2):", w[2] * hoverCost2)
            print("Cost contribution 4 (Return):", w[3] * returnCost)
            print("Cost contribution 5 (Closest Distance to Sun):", w[4] * sunCost)
            print("Cost contribution 6 (Approach velocity to NEO):", w[5] * NEOapproachVelCost)
            print("Total cost: ", cost)

        return cost
    
    def calcCostSolarExit(self):
        Penalty=100000
        ToF = self.ToF
        t_angle=self.t_angle
        t_days = self.t_days
        sailSunDistArray = self.sailSunDistArray
        sailVelocities = self.sailVelocities
        simTime=self.simTime
        dT = self.dT
        sailPos = self.sailPos
        desDist = 150 # Desired distance for solar system escape (AU) 

        AU = 1.496e11 # Astronomical Unit (m)

        # Closest distance to the sun
        sunClose = np.min(sailSunDistArray)

        # Quantify cost for too close to sun
        sunCost = 0 if sunClose > 0.1 else Penalty

        # Quantify cost for maximum distance from Sun at the end of the simulation
        sunDist = np.linalg.norm(sailPos[-1])
        distCost = abs(desDist - sunDist)/desDist
        # distCost = 0 if sunDist > desDist else Penalty

        # Calculate cost
        cost = [ sunCost, distCost, sailPos[-1]]

        return cost