from .body_data import * 
from .solarExitCost import *

def export_data(filename, time_steps, sun_data, light_sail_data, sail_normal, planet_data, num_sails):
    output_dir = "Case 2/output"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)

    with open(full_path, 'w') as f:
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

        Mercury2, Venus2, Earth2, Mars2, Jupiter2, Saturn2, Uranus2, Neptune2 = get_full_body_trajectories(5)
        planet_data2 = [Mercury2.positions, Venus2.positions, Earth2.positions, Mars2.positions, Jupiter2.positions, Saturn2.positions, Uranus2.positions, Neptune2.positions]

        # Write zones for Planet Trajectories
        planet_names2 = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        for planet2_idx, planet_name2 in enumerate(planet_names2):
            f.write(f'ZONE T="{planet_name2} 2 Trajectory"\n')
            for i in range(len(planet_data2[planet2_idx])):
                f.write(f'{planet_data2[planet2_idx][i,0]} {planet_data2[planet2_idx][i,1]} {planet_data2[planet2_idx][i,2]} 0 0 0\n')
    
    print(f"Saved output data to {full_path}")


def run_test(lightSailSolver, constant_angles, T, NumSeg, dT, numSails, bodies, dv, varSol, w, filename, save_traj):

    solver = lightSailSolver(varSol[0:dv], bodies)
    solver.runSim( constant_angles, T, NumSeg, dT)
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

    for i in range(numSails):
        solver = lightSailSolver(varSol[i*dv:(i+1)*dv], bodies)
        solver.runSim( constant_angles, T, NumSeg, dT)
        
        light_sail_data[i, :, :] = solver.sailPos
        sail_normal[i, :, :] = solver.sailNormal

    print("Generating cost for test..")
    solarExitCost(varSol, constant_angles, T, w, dT, bodies, numSails, dv, NumSeg)

    if save_traj == 1:
        print("Exporting output data...")
        export_data(filename, simTime, sunPos, light_sail_data, sail_normal, planet_data, numSails)



