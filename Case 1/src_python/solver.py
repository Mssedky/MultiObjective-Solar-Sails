import numpy as np
import os
import pandas as pd
import pandas as pd

from .solver_helpers import *

class lightSailSolver:

    def __init__(self,var,bodies):
        self.var = var
        self.bodies = bodies

    def runSim(self, desHoverTime, constant_angles, T, TOL,TOL2, MakeMovie, NumSeg, dT, traj_output_filename = 'sail_trajectory', trackNEO = 1, useEndCond = 1, NEOname = 'Bennu'):
        '''
        deHoverTime : desired hover time (days)
        constant_angles:  = 1 if you need to make the angles constant (or 0 for interpolation technique for smooth angles) 
        T : Maximum time
        TOL: Tolerance for reaching desired rendezvous distance with NEO (AU)
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
        distEarth=np.sqrt((r[0] - earthPos[len(simTime)-1,0])**2 + (r[1] - earthPos[len(simTime)-1,1])**2 + (r[2] - earthPos[len(simTime)-1,2])**2)
        earth_sail_Vel = np.linalg.norm(earthVel[len(simTime)-2,:]-sailVel)*(AU/1000)
        self.earth_sail_Vel=earth_sail_Vel
        self.finalDistEarth = distEarth
        self.reachedNEO = reachedNEO
        self.sailActive = sailActive
        self.M_sail = M_sail
        self.dT = dT
        
        # Create .dat file to send to Tecplot, also create and save dataframe for easier access to data
        if MakeMovie==1:
            print("saving data...")

            # Ensure all arrays have the same length
            min_length = min(len(self.simTime), len(self.sailActive), len(self.sailPos), len(self.earthPos))

            # Truncate arrays to the same length
            self.simTime = self.simTime[:min_length]
            self.sailActive = self.sailActive[:min_length]
            self.sailPos = self.sailPos[:min_length]
            self.sailVels = self.sailVels[:min_length]
            self.earthPos = self.earthPos[:min_length]
            self.NEOPos = self.NEOPos[:min_length]

            df = pd.DataFrame({
                "Time": self.simTime,
                "sailPosX": self.sailPos[:,0],
                "sailPosY": self.sailPos[:,1],
                "sailPosZ": self.sailPos[:,2],
                "sailVelX": self.sailVels[:,0],
                "sailVelY": self.sailVels[:,1],
                "sailVelZ": self.sailVels[:,2],
                "sailActive": self.sailActive,
                "earthPosX": self.earthPos[:,0],
                "earthPosY": self.earthPos[:,1],
                "earthPosZ": self.earthPos[:,2],
                "NEOPosX": self.NEOPos[:,0],
                "NEOPosY": self.NEOPos[:,1],
                "NEOPosZ": self.NEOPos[:,2]
            })

            # Create output directory if it doesn't exist
            output_dir = "Case 1/output"
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, traj_output_filename + ".csv"))
            
    def calcCost(self, w, TOL, TOL1, printStatements = 1):
        Penalty =1000000000
        Penalty1=100000
        distances = self.distances
        ToF = self.ToF
        endCond = self.endCond
        sailSunDistArray = self.sailSunDistArray
        finalDistEarth = self.finalDistEarth
        reachedNEO = self.reachedNEO
        earth_sail_Vel=self.earth_sail_Vel
        sailVelocities = self.sailVelocities
        simTime=self.simTime
        dT = self.dT
        desHoverTime = self.desHoverTime / dT
        M_sail=self.M_sail
        NEOVel = self.NEOVel
        AU = 1.496e11 # Astronomical Unit (m)

        # Closest distance to the sun
        sunClose = np.min(sailSunDistArray)

        # Total number of time steps light sail is within 100,000km or TOL of NEO 
        hoverTime = np.sum(np.array(distances) < TOL)

        # Find the index of the smallest distance
        min_distance_index = np.argmin(distances)

        # Ensure indices are within valid bounds
        start_index = max(0, int(min_distance_index - 0.75 * 2 * desHoverTime ))
        end_index = min(len(distances), int(min_distance_index + 0.75 * 2 * desHoverTime))

        # Calculate hoverTime2 (total number of time steps light sail is within 0.1 AU of NEO)
        hoverTime2 = np.sum(np.array(distances[start_index:end_index]) < 0.05)
        
        # Find smallest distance, corresponding approach velocity, and time step
        minDist=np.min(distances) 

        # Close distance between NEO and sail
        approxDist = 1 if minDist * Penalty < 500 else 0

        NEOapproachVelCost = 0 if reachedNEO == 1 else np.linalg.norm(NEOVel[min_distance_index] - sailVelocities[min_distance_index]) * (AU/1000) 

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
        energyCost = total_energy_to_earth*10**13 if approxDist == 1 or reachedNEO == 1 else total_energy_to_NEO*Penalty1*10**11
  

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
            print(f"Shortest time to NEO: {min_distance_time} days")
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