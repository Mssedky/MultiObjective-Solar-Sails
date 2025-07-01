import numpy as np
import os
import pandas as pd
import pandas as pd

from .solver_helpers import *

class lightSailSolver:

    def __init__(self,var,bodies):
        self.var = var
        self.bodies = bodies

    def runSim(self, constant_angles, T, NumSeg, dT):
        '''
        deHoverTime : desired hover time (days)
        constant_angles:  = 1 if you need to make the angles constant (or 0 for interpolation technique for smooth angles) 
        T : Maximum time
        NumSeg: Number of time segments in trajectory plan
        dT: Time step (Days)
        trackNEO: if trackNEO=1, track the NEO, if trackNEO=0, don't track NEO
        useEndCond: stop simulation when end condition is reached if useEndCond = 1
        '''

        var = self.var
        bodies = self.bodies

        Earth = find_body(bodies, 'Earth')
        Mercury = find_body(bodies, 'Mercury')
        Venus = find_body(bodies, 'Venus')
        Mars = find_body(bodies, 'Mars')
        Jupiter = find_body(bodies, 'Jupiter')
        Saturn = find_body(bodies, 'Saturn')
        Uranus = find_body(bodies, 'Uranus')
        Neptune = find_body(bodies, 'Neptune')
        Pluto = find_body(bodies, 'Pluto')

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
        SlingshotDist = 1000000*1000/AU # Distance from Earth's surface to location after slingshotting away from Earth's gravity (1,000,000km converted to AU)
        rInit=Radius_Earth+SlingshotDist # Initial distance of Sail from Earth's center (AU)
        beta = 0.16 # Lightness number (dimensionless) - see Mclnnes eq. 2.25a p. 40 and BLISS paper

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
        sailVels = np.empty((0,3))
        sunPos = np.empty((0, 3))
        sailNormal = np.empty((0,3)) 
        sailFlightPos = np.empty((0,3))
        sailVelocities = []
        sailSunDistArray = []
        NEO_sail_Vel=[]
        distances = []
        simTime=[]
        ToF=[]
        alphaG=[]
        gammaG=[]

        sailActive=[] #track status: 0=not in flight, 1=in flight/active

        time_segments = calc_time_segments(var[3:NumSeg+3])
        cone_angle_var = var[NumSeg+3:2*NumSeg+3]
        clock_angle_var = var[2*NumSeg+3:3*NumSeg+3]
        segmentsClocks, segmentsCones = create_angle_functions(time_segments, clock_angle_var, cone_angle_var, degree)

        # Main loop
        while(t_days < T): 
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
                sailVels = np.vstack((sailVels, v))

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


                # Forward Euler velocity solver
                if np.cos(alpha) <= 0:  # sail perpendicular to or away from the sun -> no solar force
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
                sailNormal = np.vstack((sailNormal, n / np.linalg.norm(n)))

                sailVel = v
                sailPos = np.vstack((sailPos,r))
                sailFlightPos = np.vstack((sailFlightPos,r))
                sailVelocities.append(v)  # Store velocity for energy calculation
                sailVels = np.vstack((sailVels, v))
                
                #Calculate distance between sail and sun
                sailSunDist = np.linalg.norm(r)
                sailSunDistArray=np.append(sailSunDistArray,sailSunDist) 

                t_angle=t_angle+dT
                ToF=np.append(ToF,t_angle)
                alphaG=np.append(alphaG,alpha)
                gammaG=np.append(gammaG,gamma)
                sailActive.append(1)
                

            t=t+dt # time progression in seconds
            t_days=dT+t_days

        
        
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
        self.sunPos = sunPos
        self.sailNormal = sailNormal
        self.sailFlightPos=sailFlightPos
        self.sailVels = sailVels
        self.distances = distances
        self.simTime=simTime
        self.t_angle=t_angle
        self.ToF=ToF
        self.alphaG=alphaG
        self.gammaG=gammaG
        self.endCond = endCond
        self.t_days = t_days
        self.sailVelocities = sailVelocities
        self.reachedNEO = reachedNEO
        self.sailActive = sailActive
        self.M_sail = M_sail
        self.dT = dT
        
            
    def calcCostSolarExit(self):
        Penalty=100000
        sailSunDistArray = self.sailSunDistArray
        sailPos = self.sailPos
        desDist = 100 # Desired distance for solar system escape (AU) 

        AU = 1.496e11 # Astronomical Unit (m)

        # Closest distance to the sun
        sunClose = np.min(sailSunDistArray)

        # Quantify cost for too close to sun
        sunCost = 0 if sunClose > 0.2 else Penalty

        # Quantify cost for maximum distance from Sun at the end of the simulation
        sunDist = np.linalg.norm(sailPos[-1])
        distCost = abs(desDist - sunDist)/desDist

        # Calculate cost
        cost = [ sunCost, distCost, sailPos[-1]]

        return cost