import numpy as np
from scipy.optimize import fsolve

def orbital_elements_to_cartesian(a, e, i, omega, Omega, nu):
    p = a * (1 - e**2)  # Semi-latus rectum
    r = p / (1 + e * np.cos(nu))  # Radius
    x = r * (np.cos(Omega) * np.cos(omega + nu) - np.sin(Omega) * np.sin(omega + nu) * np.cos(i))
    y = r * (np.sin(Omega) * np.cos(omega + nu) + np.cos(Omega) * np.sin(omega + nu) * np.cos(i))
    z = r * np.sin(omega + nu) * np.sin(i)  
    return x, y, z

def calcTrueAnomaly(semi_major_axis, eccentricity, mean_anomaly, t):
    AU = 1.496e11 #AU in m
    G = 6.67430e-11/(AU**3)  # Gravitational constant (AU^3/kg/s^2)
    M_sun = 1.989e30  # Mass of the Sun (kg)

    # Calculate mean anomaly
    mean_angular_velocity = np.sqrt(G * M_sun / semi_major_axis**3)  # Angular velocity (radians/second)
    mean_anomaly += mean_angular_velocity * t  # Mean anomaly (radians)

    # Define Kepler's equation to solve for eccentric anomaly
    def kepler_equation(eccentric_anomaly):
        return eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly

    # Initial guess for eccentric anomaly
    initial_guess = mean_anomaly

    # Use scipy's fsolve to find the root of Kepler's equation
    eccentric_anomaly = fsolve(kepler_equation, initial_guess)[0]  # Get the first (and only) solution

    # Calculate true anomaly
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + eccentricity) * np.sin(eccentric_anomaly / 2),
                                  np.sqrt(1 - eccentricity) * np.cos(eccentric_anomaly / 2))

    return true_anomaly


def find_body(bodies, name):
    for body in bodies:
        if body.name == name:
            return body
    return None

class orbitingObject:
    def __init__(self, a, e, i, omega, Omega, nuMean, mass, name, positions, velocities):
        '''
        a = semi_major_axis # Semi-major axis of Earth's orbit (AU)
        e = eccentricity # Eccentricity of Earth's orbit
        i = inclination # Inclination (radians)
        omega = periapsis_arg # Argument of periapsis (radians)
        Omega = longitude_asc_node # Longitude of ascending node (radians)
        nuMean=mean_anomaly # Approximate mean anomaly (radians)
        '''
        self.a = a
        self.e = e
        self.i = i
        self.omega = omega
        self.Omega = Omega
        self.nuMean = nuMean
        self.mass = mass
        self.name = name
        self.positions = positions
        self.velocities = velocities

# Function to solve for the planets and NEO's location a priori
def calcCelestialTraj(body, dT, T, tTraj = 1):
    t = 0  # seconds
    if tTraj:
        dT_sim = 0.1  # days
    else:
        dT_sim = 1

    dt = dT_sim * 24 * 60 * 60  # Converting dT to seconds
    t_days = 0

    bodyPos = np.empty((0, 3))
    bodyVel = np.empty((0, 3))
    positions = []
    velocities = []

    interval_counter = 0  # Counter to track intervals of dT

    while t_days < T:
        # Calculate true anomaly
        nu = calcTrueAnomaly(body.a, body.e, body.nuMean, t)

        # Calculate Cartesian coordinates of body
        xN, yN, zN = orbital_elements_to_cartesian(body.a, body.e, body.i, body.omega, body.Omega, nu)
        bodyPos = np.vstack((bodyPos, [xN, yN, zN]))

        # Store positions at intervals of dT
        if interval_counter % int(dT / dT_sim) == 0:
            positions.append([xN, yN, zN])

            # Calculate and store velocity if there are at least two position points
            if len(bodyPos) > 1:
                vel = (bodyPos[-1] - bodyPos[-2]) / dt
                bodyVel = np.vstack((bodyVel, vel))
                velocities.append(vel)

        t = t + dt
        t_days = t_days + dT_sim
        interval_counter += 1

    body.positions = np.array(positions)
    body.velocities = np.array(velocities)

    return body