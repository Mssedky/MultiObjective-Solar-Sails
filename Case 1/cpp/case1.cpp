#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <memory>
#include<functional>
#include <limits>
#include <iomanip>
#include <chrono>

// Constants
const double AU = 1.496e11;  // Astronomical Unit (m)
const double G = 6.67430e-11;  // Gravitational constant (m^3/kg/s^2)
const double M_sun = 1.989e30;  // Mass of the Sun (kg)
const double PI = 3.14159265359;


class Vector3D {
public:
    double x, y, z;
    
    Vector3D() : x(0.0), y(0.0), z(0.0) {}
    Vector3D(double x, double y, double z) : x(x), y(y), z(z) {}
    
    Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3D operator-(const Vector3D& other) const {
        return Vector3D(x - other.x, y - other.y, z - other.z);
    }
    
    Vector3D operator*(double scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }
    
    double dot(const Vector3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    Vector3D cross(const Vector3D& other) const {
        return Vector3D(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    double norm() const {
        return sqrt(x * x + y * y + z * z);
    }
    
    Vector3D normalize() const {
        double n = norm();
        return n > 0 ? Vector3D(x/n, y/n, z/n) : Vector3D();
    }
};

// Function to convert orbital elements to Cartesian coordinates
Vector3D orbitalElementsToCartesian(double a, double e, double i, double omega, 
                                   double Omega, double nu) {
    double p = a * (1 - e * e);  // Semi-latus rectum
    double r = p / (1 + e * cos(nu));  // Radius
    
    double x = r * (cos(Omega) * cos(omega + nu) - sin(Omega) * sin(omega + nu) * cos(i));
    double y = r * (sin(Omega) * cos(omega + nu) + cos(Omega) * sin(omega + nu) * cos(i));
    double z = r * sin(omega + nu) * sin(i);
    
    return Vector3D(x, y, z);
}

// Function to calculate true anomaly
double calcTrueAnomaly(double semi_major_axis, double eccentricity, 
                      double mean_anomaly, double t) {
    double mean_angular_velocity = sqrt(G * M_sun / (AU * AU * AU) / (semi_major_axis * semi_major_axis * semi_major_axis));
    mean_anomaly += mean_angular_velocity * t;
    
    // Solve Kepler's equation iteratively (Newton-Raphson)
    double E = mean_anomaly;  // Initial guess
    for (int iter = 0; iter < 1000; ++iter) {
        double f = E - eccentricity * sin(E) - mean_anomaly;
        double df = 1 - eccentricity * cos(E);
        double dE = f / df;
        E -= dE;
        if (abs(dE) < 1e-12) break;
    }
    
    // Convert eccentric anomaly to true anomaly
    double true_anomaly = 2 * atan2(sqrt(1 + eccentricity) * sin(E / 2),
                                   sqrt(1 - eccentricity) * cos(E / 2));
    
    return true_anomaly;
}


class OrbitingObject {
public:
    double a, e, i, omega, Omega, nuMean, mass;
    std::string name;
    std::vector<Vector3D> positions;
    std::vector<Vector3D> velocities;
    
    OrbitingObject(double a, double e, double i, double omega, double Omega, 
                   double nuMean, double mass, const std::string& name)
        : a(a), e(e), i(i), omega(omega), Omega(Omega), 
          nuMean(nuMean), mass(mass), name(name) {}
};

// Function to calculate celestial trajectory
void calcCelestialTraj(OrbitingObject& body, double dT, double T) {
    double t = 0;  // seconds
    double dT_sim = 0.1;  // days
    double dt = dT_sim * 24 * 60 * 60;  // Converting dT to seconds
    double t_days = 0;
    
    int interval_counter = 0;
    int interval_size = static_cast<int>(dT / dT_sim);
    
    while (t_days < T) {
        // Calculate true anomaly
        double nu = calcTrueAnomaly(body.a, body.e, body.nuMean, t);
        
        // Calculate Cartesian coordinates
        Vector3D pos = orbitalElementsToCartesian(body.a, body.e, body.i, 
                                                 body.omega, body.Omega, nu);
        
        // Store positions at intervals of dT
        if (interval_counter % interval_size == 0) {
            body.positions.push_back(pos);
            
            // Calculate and store velocity if there are at least two position points
            if (body.positions.size() > 1) {
                Vector3D vel = (body.positions.back() - body.positions[body.positions.size()-2]) * (1.0 / dt);
                body.velocities.push_back(vel);
            }
        }
        
        t += dt;
        t_days += dT_sim;
        interval_counter++;
    }
}

void printDesignVariables(const std::vector<double>& var, int NumSeg) {
    std::cout << std::fixed << std::setprecision(8);  
    
    std::cout << "The design variables are:" << std::endl;
    std::cout << "[";
    
    for (size_t i = 0; i < var.size(); ++i) {
        std::cout << var[i];
        
        if (i < var.size() - 1) {
            std::cout << ", ";
        }

        if ((i + 1) % 6 == 0 && i < var.size() - 1) {
            std::cout << std::endl << " "; 
        }
    }
    
    std::cout << "]" << std::endl;
    
    // Reset to default precision
    std::cout << std::resetiosflags(std::ios::fixed | std::ios::floatfield);
}

// Helper functions for angle interpolation
std::vector<double> calcTimeSegments(const std::vector<double>& time_var) {
    std::vector<double> segments;
    segments.push_back(0);
    double cumsum = 0;
    for (double t : time_var) {
        cumsum += t;
        segments.push_back(cumsum);
    }
    return segments;
}

std::pair<double, double> parseAngles(const std::vector<double>& time_segments, 
                                     const std::vector<double>& cone_angles, 
                                     const std::vector<double>& clock_angles, 
                                     double t_days) {
    auto it = std::upper_bound(time_segments.begin(), time_segments.end(), t_days);
    int ind = std::distance(time_segments.begin(), it);
    
    if (ind < time_segments.size() - 1) {
        return {cone_angles[ind], clock_angles[ind]};
    } else {
        return {0, 0};
    }
}


// Structure to hold polynomial data for a segment
struct AngleSegment {
    std::vector<double> times;
    std::vector<double> values;
    std::function<double(double)> poly;
    
    AngleSegment(const std::vector<double>& t, const std::vector<double>& v, 
                 std::function<double(double)> p) 
        : times(t), values(v), poly(p) {}
};


std::function<double(double)> lagrange_interpolation(const std::vector<double>& x_points, 
                                                   const std::vector<double>& y_points) {
    return [x_points, y_points](double x) -> double {
        double result = 0.0;
        int n = x_points.size();
        
        for (int i = 0; i < n; i++) {
            double term = y_points[i];
            
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    term *= (x - x_points[j]) / (x_points[i] - x_points[j]);
                }
            }
            
            result += term;
        }
        
        return result;
    };
}


std::pair<std::vector<AngleSegment>, std::vector<AngleSegment>> 
create_angle_functions(const std::vector<double>& time_segments,
                      const std::vector<double>& clock_angles,
                      const std::vector<double>& cone_angles,
                      int degree) {
    
    std::vector<AngleSegment> segments_clocks;
    std::vector<AngleSegment> segments_cones;
    
    int i = 0;
    int n = clock_angles.size();
    
    while (i < n - 1) {
        // Determine the number of points available to fit a polynomial
        int points_remaining = n - i;
        int degree_to_fit = std::min(degree, points_remaining - 1);
        
        // Cap the degree to avoid instability
        if (degree_to_fit > 4) {
            degree_to_fit = 4;  // Restrict to degree 4 to avoid overfitting
        }
        
        // Number of points needed to fit a polynomial of degree `degree_to_fit`
        int points_needed = degree_to_fit + 1;
        
        // Select the points for this segment
        std::vector<double> clocks(clock_angles.begin() + i, 
                                  clock_angles.begin() + i + points_needed);
        std::vector<double> cones(cone_angles.begin() + i, 
                                 cone_angles.begin() + i + points_needed);
        std::vector<double> times(time_segments.begin() + i, 
                                 time_segments.begin() + i + points_needed);
        
        // Fit an exact polynomial to clock angles using Lagrange interpolation
        auto poly_clocks = lagrange_interpolation(times, clocks);
        segments_clocks.emplace_back(times, clocks, poly_clocks);
        
        // Fit an exact polynomial to cone angles using Lagrange interpolation
        auto poly_cones = lagrange_interpolation(times, cones);
        segments_cones.emplace_back(times, cones, poly_cones);
        
        i += degree_to_fit;
    }
    
    return std::make_pair(segments_clocks, segments_cones);
}


double find_value_at_time(const std::vector<AngleSegment>& angles,
                         const std::vector<double>& time_segments,
                         double t_days) {
    
    
    auto it = std::upper_bound(time_segments.begin(), time_segments.end(), t_days);
    int ind = std::distance(time_segments.begin(), it);
    
    if (ind < static_cast<int>(time_segments.size()) - 1) {
        for (const auto& segment : angles) {
            double min_time = *std::min_element(segment.times.begin(), segment.times.end());
            double max_time = *std::max_element(segment.times.begin(), segment.times.end());
            
            if (min_time <= t_days && t_days <= max_time) {
                return segment.poly(t_days);
            }
        }
    }
    
    return 0.0;
}


Vector3D initialPos(double xE, double yE, double rInit) {
    double x[2] = {xE + rInit, yE};
    
    // Parameters
    const int maxfev = 200;
    const double xtol = 1.0e-12;
    const double ftol = 1.0e-12;
    
    // Working arrays
    double fvec[2];
    double fjac[4]; 
    
    // Machine precision
    double eps = std::sqrt(2.22e-16);
    
    // Function evaluation
    auto fcn = [&](const double* x_val, double* f_val) {
        double dx = x_val[0] - xE;
        double dy = x_val[1] - yE;  
        f_val[0] = dx * xE + dy * yE;                    
        f_val[1] = dx * dx + dy * dy - rInit * rInit;    
    };
    
    int nfev = 0;
    bool converged = false;
    
    for (int iter = 0; iter < maxfev && !converged; iter++) {
        fcn(x, fvec);
        nfev++;
        
        // Check function convergence
        double fnorm = std::sqrt(fvec[0] * fvec[0] + fvec[1] * fvec[1]);
        if (fnorm <= ftol) {
            converged = true;
            break;
        }
        
        // Compute Jacobian analytically
        double dx = x[0] - xE;
        double dy = x[1] - yE;
        
        // Analytical Jacobian:
        // f1 = dx * xE + dy * yE
        // df1/dx = xE, df1/dy = yE
        // f2 = dx² + dy² - rInit²  
        // df2/dx = 2*dx, df2/dy = 2*dy
        fjac[0] = xE;           // df1/dx
        fjac[1] = yE;           // df1/dy
        fjac[2] = 2.0 * dx;     // df2/dx
        fjac[3] = 2.0 * dy;     // df2/dy
        
        // Solve linear system: fjac * p = -fvec using Cramer's rule
        double det = fjac[0] * fjac[3] - fjac[1] * fjac[2];
        
        if (std::abs(det) < 1e-14) {
            // Singular Jacobian - try a different approach
            // Use steepest descent with small step
            double grad_norm = std::sqrt(fjac[0]*fjac[0] + fjac[1]*fjac[1] + 
                                        fjac[2]*fjac[2] + fjac[3]*fjac[3]);
            if (grad_norm > 1e-14) {
                double alpha = 0.1 * std::min(xtol, ftol) / grad_norm;
                x[0] -= alpha * (fjac[0] * fvec[0] + fjac[2] * fvec[1]);
                x[1] -= alpha * (fjac[1] * fvec[0] + fjac[3] * fvec[1]);
            }
        } else {
            // Newton step: p = -J^(-1) * f
            double p[2];
            p[0] = (-fvec[0] * fjac[3] + fvec[1] * fjac[1]) / det;
            p[1] = ( fvec[0] * fjac[2] - fvec[1] * fjac[0]) / det;
            
            // Apply step 
            x[0] += p[0];
            x[1] += p[1];
            
            // Check step convergence
            double pnorm = std::sqrt(p[0] * p[0] + p[1] * p[1]);
            if (pnorm <= xtol) {
                converged = true;
            }
        }
    }
    
    return Vector3D(x[0], x[1], 0.0);
}

std::shared_ptr<OrbitingObject> find_body(const std::vector<std::shared_ptr<OrbitingObject>>& bodies,
                                          const std::string& name){
    
    for (const auto& body:bodies){
        if(body->name == name){
            return body;
        }
    }

    return nullptr;
}


class LightSailSolver {
private:
    std::vector<double> var;
    std::vector<std::shared_ptr<OrbitingObject>> bodies;
    
public:
    std::vector<Vector3D> sailPos;
    std::vector<Vector3D> earthPos;
    std::vector<Vector3D> earthVel;
    std::vector<Vector3D> NEOPos;
    std::vector<Vector3D> NEOVel;
    std::vector<Vector3D> sunPos;
    std::vector<Vector3D> sailNormal;
    std::vector<Vector3D> sailFlightPos;
    std::vector<Vector3D> sailVelocities;
    std::vector<Vector3D> sailVels;
    std::vector<double> sailSunDistArray;
    std::vector<double> distances;
    std::vector<double> simTime;
    std::vector<double> ToF;
    std::vector<double> alphaG;
    std::vector<double> gammaG;
    std::vector<int> sailActive;
    double finalDistEarth;
    double earthSailVel;
    double desHoverTime;
    double t_angle;
    double t_days;
    double M_sail;
    double dT;
    double distNEO = 100; // initial hypothetical distance
    double distEarth;
    int reachedNEO;
    int endCond;
    bool trackNEO = true;
    
    LightSailSolver(const std::vector<double>& variables, 
                   const std::vector<std::shared_ptr<OrbitingObject>>& celestialBodies)
        : var(variables), bodies(celestialBodies), reachedNEO(0), endCond(0),
          finalDistEarth(0), earthSailVel(0), t_angle(0), t_days(0), M_sail(14.6e-3) {}
    
    void runSim(double desHoverTime, bool constant_angles, double T, 
               double TOLNEO, double TOLEarth, int NumSeg, double dT_param) {
        
        this->desHoverTime = desHoverTime;
        this->dT = dT_param;
        
        // Constants
        double dt = dT_param * 24 * 60 * 60;  // Convert dT to seconds
        double beta = 0.16;  // Lightness number
        double Radius_Earth = 6378000 / AU;  // Earth's radius (AU)
        double SlingshotDist = 1000000 * 1000 / AU;  // Distance from Earth (AU)
        double rInit = Radius_Earth + SlingshotDist;  // Initial distance from Earth
        
        // Gravitational parameters
        double muSun = G * M_sun / (AU * AU * AU);
        double muEarth = G * 5.97219e24 / (AU * AU * AU);  // Earth mass
        
        // Extract variables
        int degree = static_cast<int>(var[0]);
        double initial_launch = var[1];
        double vInit = var[2];
        
        // Extract time segments and angles
        std::vector<double> time_var(var.begin() + 3, var.begin() + 3 + NumSeg);
        std::vector<double> cone_angle_var(var.begin() + 3 + NumSeg, var.begin() + 3 + 2 * NumSeg);
        std::vector<double> clock_angle_var(var.begin() + 3 + 2 * NumSeg, var.begin() + 3 + 3 * NumSeg);
        
        std::vector<double> time_segments = calcTimeSegments(time_var);
        
        // Create angle functions
        auto [segmentsClocks, segmentsCones] = create_angle_functions(time_segments, clock_angle_var, cone_angle_var, degree);
        
        auto Earth = find_body(bodies, "Earth");
        auto NEO = find_body(bodies, "Bennu");
        
        earthPos = Earth->positions;
        earthVel = Earth->velocities;
        NEOPos = NEO->positions;
        NEOVel = NEO->velocities;

        if (!Earth || !NEO) {
            std::cerr << "Error: Required celestial bodies not found!" << std::endl;
            return;
        }
        
        // Initialize variables
        double t = 0.0;
        t_days = 0;
        t_angle = 0;
        reachedNEO = 0;
        endCond = 0;
        int setInitDir = 1;
        
        Vector3D r, v;
        Vector3D p(0.0, 0.0, 1.0);  // Normal vector to Earth's orbit
        
        // Main simulation loop
        while (t_days < T && endCond == 0) {
            // Store simulation time
            simTime.push_back(t_days);
            
            // Sun position (always at origin)
            sunPos.push_back(Vector3D(0.0, 0.0, 0.0));
            
            // Pre-launch phase
            if (t_days < initial_launch) {
                // Calculate sail position relative to Earth
                Vector3D initialSailPos = initialPos(earthPos[simTime.size()-1].x, earthPos[simTime.size() - 1].y, rInit);
                sailPos.push_back(initialSailPos);
                sailNormal.push_back(Vector3D(0.0, 0.0, 0.0));
                r = initialSailPos;

                // Determine which side of earth the solar sail is on
                Vector3D earthVelCurrent = (t == 0.0) ? earthVel[simTime.size()] : earthVel[simTime.size() - 1];
                Vector3D sailEarthDiff = sailPos.back() - earthPos[simTime.size() - 1];
                
                double dotProduct = earthVelCurrent.dot(sailEarthDiff);
                int dirIndicator = (dotProduct > 0) ? 1 : 0;

                 // Set the sail to the correct side of Earth
                if (setInitDir != dirIndicator) {
                    sailPos.back() =  earthPos[simTime.size() - 1] * 2.0 - sailPos.back();
                    r = sailPos.back();
                }
                
                // Set initial velocity
                Vector3D r0 = (r - earthPos[simTime.size()-1]).normalize();
                v = r0 * vInit;
                sailActive.push_back(0);
                sailVels.push_back(v);
            }
            // Post-launch phase
            else {
                // Calculate cone and clock angles (matching Python logic exactly)
                double alpha, gamma;

                if (constant_angles) {
                    // Use step function (parseAngles)
                    auto angles = parseAngles(time_segments, cone_angle_var, clock_angle_var, t_days - initial_launch);
                    alpha = angles.first;
                    gamma = angles.second;
                } else {
                    // Use Lagrange polynomial interpolation functions
                    alpha = find_value_at_time(segmentsCones, time_segments, t_days);
                    gamma = find_value_at_time(segmentsClocks, time_segments, t_days);
                }
                
                // Clip angles to allowable range
                alpha = std::max(-PI/2.0, std::min(PI/2.0, alpha));
                gamma = std::max(-PI, std::min(PI, gamma));
                
                if(t_days == initial_launch){
                    p = Vector3D(0.0,0.0,1.0); // normal vector to Earth's orbit
                }else{
                    p = r.cross(v);
                }

                // Calculate sail normal vector 
                Vector3D r_unit = r * (1.0 /r.norm());
                Vector3D p_unit = p*(1.0 /p.norm());
                Vector3D tangent = p_unit.cross(r_unit);  // Tangential direction
                
                Vector3D n1 = r_unit * cos(alpha);  // Radial component
                Vector3D n2 = p_unit * (cos(gamma) * sin(alpha));  // Out-of-plane component
                Vector3D n3 = tangent * (sin(gamma) * sin(alpha));  // In-plane tangential component
                
                Vector3D n = n1 + n2 + n3;
                Vector3D n_unit = n * (1.0 / n.norm());

                // Forward Euler integration 
                double cos_alpha = cos(alpha);

                if (cos_alpha <= 0 ){
                    v = v -  (r * (1.0 / (r.norm() * r.norm() * r.norm()))) * dt * muSun 
                    -  (earthPos[simTime.size()-1] - r) * dt * muEarth * 
                    (1 / ((earthPos[simTime.size()-1] - r).norm() * (earthPos[simTime.size()-1] - r).norm() * (earthPos[simTime.size()-1] - r).norm() ));
                }else{
                    v = v + (n * (1.0/n.norm())) * dt * beta * (muSun / (r.norm()*r.norm())) * (cos_alpha * cos_alpha)
                    -  (r * (1.0 / (r.norm() * r.norm() * r.norm()))) * dt * muSun 
                    -  (earthPos[simTime.size()-1] - r) * dt * muEarth * 
                    (1.0 / ((earthPos[simTime.size()-1] - r).norm() * (earthPos[simTime.size()-1] - r).norm() * (earthPos[simTime.size()-1] - r).norm() ));
                }

                r = r + v * dt;
                
                // Store results
                sailPos.push_back(r);
                sailNormal.push_back(n_unit);
                sailFlightPos.push_back(r);
                sailVelocities.push_back(v);
                sailVels.push_back(v);

                // Calculate distance to sun
                double sailSunDist = r.norm();
                sailSunDistArray.push_back(sailSunDist);
                
                if (trackNEO){
                    Vector3D neoPos = NEOPos[simTime.size()-1];
                    distNEO = (r - neoPos).norm();
                    distances.push_back(distNEO);
                    // std::cout<<"DistNEO 1: "<<distNEO << std::endl;
                }
                
                t_angle += dT_param;
                ToF.push_back(t_angle);
                alphaG.push_back(alpha);
                gammaG.push_back(gamma);
                sailActive.push_back(1);
            }

            // sailVels.push_back(v);
            t += dt;
            t_days += dT;

            if (trackNEO){
                if (reachedNEO == 0){
                    if(distNEO < TOLNEO){
                        // std::cout<<"DistNEO 2: "<<distNEO << std::endl;
                        reachedNEO = 1;
                    }
                }else{
                   distEarth = (r - earthPos[simTime.size()-1]).norm();
                   if(distEarth<TOLEarth){
                    endCond = 1;
                   }
                }
            }
        }
        // Set final distance to Earth if simulation completed
        if (!earthPos.empty() && !sailPos.empty()) {
            finalDistEarth = (sailPos.back() - earthPos.back()).norm();
        }
    }
    
    double calcCost(const std::vector<double>& w, double TOLNEO, double TOLEarth, bool printStatements = false) {
        double Penalty = 1000000000;
        double Penalty1 = 100000;
        
        // Convert desired hover time from days to time steps
        double desHoverTimeSteps = desHoverTime / dT;

        // Closest distance to the sun
        double sunClose = *std::min_element(sailSunDistArray.begin(), sailSunDistArray.end());
        
        // Total number of time steps light sail is within TOL of NEO
        int hoverTime = std::count_if(distances.begin(), distances.end(), [TOLNEO](double d) { return d < TOLNEO; });

        // Find the index of the smallest distance
        auto min_it = std::min_element(distances.begin(), distances.end());
        size_t min_distance_index = std::distance(distances.begin(), min_it);
        double minDist = *min_it;
        
        // Calculate hover time in specific window around closest approach
        int start_index = std::max(0, static_cast<int>(min_distance_index - 0.75 * 2 * desHoverTime));
        int end_index = std::min(static_cast<int>(distances.size()), 
                                static_cast<int>(min_distance_index + 0.75 * 2 * desHoverTime));
        

        double hoverTime2 = 0;
        for (int i = start_index; i < end_index; ++i) {
            if (distances[i] < 0.05) hoverTime2++;
        }
        
        
        // Close distance between NEO and sail
        int approxDist = (minDist * Penalty < 500) ? 1 : 0;

        // NEO approach velocity cost
        double NEOapproachVelCost = 0;
        if (min_distance_index < sailVelocities.size() && min_distance_index < NEOPos.size()) {
            // Get NEO velocity at closest approach time
            Vector3D neoVel(0, 0, 0);  // Simplified - NEO velocity is much slower than sail
            if (min_distance_index < NEOVel.size()) {
                neoVel = NEOVel[min_distance_index];
            }
            
            Vector3D relativeVel = sailVelocities[min_distance_index] - neoVel;
            NEOapproachVelCost = relativeVel.norm() * (AU / 1000);  // Convert to km/s
        }
        
        // Distance norm to NEO
        double distNorm = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();

        // Get the corresponding time step
        double min_distance_time = ToF[min_distance_index];


        // Cost components
        double hoverCost = reachedNEO == 1 ? 
            std::abs(desHoverTimeSteps - hoverTime) / desHoverTimeSteps : 
            minDist * Penalty;
            
        double hoverCost2 = minDist < 0.1 ? 
            std::abs(desHoverTimeSteps * 2 - hoverTime2) / (2 * desHoverTimeSteps) : 
            minDist * Penalty;
            
        double returnCost = endCond == 1 ? 0 : finalDistEarth;
        double sunCost = sunClose > 0.25 ? 0 : Penalty;

        // Energy cost calculation
        double total_energy_to_NEO = 0;
        for (int i = 0; i < min_distance_index; ++i) {
            double vel_norm = sailVelocities[i].norm();
            total_energy_to_NEO += 0.5 * M_sail * vel_norm * vel_norm;
        }

        double total_energy = 0.0;
        for (const auto& vel : sailVelocities) {
            double vel_norm = vel.norm();
            total_energy += 0.5 * M_sail * vel_norm * vel_norm;
        }

        double total_energy_to_earth = 0;
        for (int i = min_distance_index; i < sailVelocities.size(); ++i) {
            double speed = sailVelocities[i].norm();
            total_energy_to_earth += 0.5 * M_sail * speed * speed;
        }
        
        double energyCost = (approxDist == 1 || reachedNEO == 1) ? 
                           total_energy_to_earth * 1e13 : 
                           total_energy_to_NEO * Penalty1 * 1e11;
        
        // Calculate total cost
        double cost = w[0] * energyCost +
                     w[1] * hoverCost +
                     w[2] * hoverCost2 +
                     w[3] * returnCost +
                     w[4] * sunCost +
                     w[5] * NEOapproachVelCost;
        
        if (printStatements) {
            std::cout << "Shortest time to NEO: " << (ToF.empty() ? 0 : ToF[min_distance_index]) << " days" << std::endl;
            std::cout << "Total flight time: " << (simTime.empty() ? 0 : simTime.back()) << " days" << std::endl;
            std::cout << "Reached NEO: " << (reachedNEO == 1 ? "Yes" : "No") << std::endl;
            std::cout << "Returned to Earth: " << (endCond == 1 ? "Yes" : "No") << std::endl;
            std::cout << "Cost contribution 1 (Total energy): " << w[0] * energyCost << std::endl;
            std::cout << "Cost contribution 2 (Hover Time): " << w[1] * hoverCost << std::endl;
            std::cout << "Cost contribution 3 (Hover Time 2): " << w[2] * hoverCost2 << std::endl;
            std::cout << "Cost contribution 4 (Return): " << w[3] * returnCost << std::endl;
            std::cout << "Cost contribution 5 (Closest Distance to Sun): " << w[4] * sunCost << std::endl;
            std::cout << "Cost contribution 6 (Approach velocity to NEO): " << w[5] * NEOapproachVelCost << std::endl;
            std::cout << "Total cost: " << cost << std::endl;
        }
        
        return cost;
    }
};


double lightSailCost(const std::vector<double>& var, double desHoverTime, bool constant_angles, 
                    double T, const std::vector<double>& w, double TOLNEO, double TOLEarth, int NumSeg,
                    double dT, const std::vector<std::shared_ptr<OrbitingObject>>& bodies) {
    
    LightSailSolver solver(var, bodies);
    solver.runSim(desHoverTime, constant_angles, T, TOLNEO, TOLEarth, NumSeg, dT);
    return solver.calcCost(w, TOLNEO, TOLEarth, false);  // Set to false to avoid too much output during optimization
}


// Function to save simulation results (sail trajectory)
void saveSailTrajectoryToFile(const LightSailSolver& solver, const std::string& filename = "sail_trajectory_cpp.txt") {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing!" << std::endl;
        return;
    }
    
    // Set precision for output
    file << std::fixed << std::setprecision(12);
    
    // Write header
    file << "# Light Sail Trajectory" << std::endl;
    file << "# Units: Positions in AU, Velocities in AU/s, Distances in AU, Time in days" << std::endl;
    file << "# Format: Time(days) Sail_Pos_X Sail_Pos_Y Sail_Pos_Z Sail_Vel_X Sail_Vel_Y Sail_Vel_Z Active Alpha Gamma" << std::endl;
    file << "#" << std::endl;
    
    size_t max_size = std::min({solver.simTime.size(), solver.sailPos.size(), 
                               solver.sailVels.size()});

    
    for (size_t i = 0; i < max_size; ++i) {
        file << solver.simTime[i] << " ";
        
        // Sail position
        if (i < solver.sailPos.size()) {
            file << solver.sailPos[i].x << " " << solver.sailPos[i].y << " " << solver.sailPos[i].z << " ";
        } else {
            file << "0 0 0 ";
        }
        
        // Sail velocity
        if (i < solver.sailVels.size()) {
            file << solver.sailVels[i].x << " " << solver.sailVels[i].y << " " << solver.sailVels[i].z << " ";
        } else {
            file << "0 0 0 ";
        }
        
        if (i < solver.sailVels.size()) {
            file << solver.sailVels[i].x << " " << solver.sailVels[i].y << " " << solver.sailVels[i].z << " ";
        } else {
            file << "0 0 0 ";
        }

        // Sail active status
        if (i < solver.sailActive.size()) {
            file << solver.sailActive[i] << " ";
        } else {
            file << "0 ";
        }
        
        // Angles
        if (i < solver.alphaG.size()) {
            file << solver.alphaG[i] << " ";
        } else {
            file << "0 ";
        }
        
        if (i < solver.gammaG.size()) {
            file << solver.gammaG[i];
        } else {
            file << "0";
        }
        
        file << std::endl;
    }
    
    file.close();
    std::cout << "Sail trajectory saved to " << filename << std::endl;
    std::cout << "Data points: " << max_size << std::endl;
}

// Genetic Algorithm class
class GeneticAlgorithm {
private:
    int population_size;
    int num_parents;
    int num_children;
    int max_generations;
    double tolerance;
    std::vector<double> lb, ub;  // Lower and upper bounds
    std::mt19937 rng;
    
public:
    GeneticAlgorithm(int pop_size, int parents, int children, int max_gen, 
                    double tol, const std::vector<double>& lower_bounds, 
                    const std::vector<double>& upper_bounds, 
                    unsigned int seed = 0) 
        : population_size(pop_size), num_parents(parents), num_children(children),
          max_generations(max_gen), tolerance(tol), lb(lower_bounds), ub(upper_bounds) {
        
        // Initialize random number generator with seed
        if (seed == 0) {
            // Use random device if no seed specified (default behavior)
            rng.seed(std::random_device{}());
            std::cout << "Using random seed from random_device" << std::endl;
        } else {
            // Use specified seed for reproducible results
            rng.seed(seed);
            std::cout << "Using fixed seed: " << seed << std::endl;
        }
    }
    
    std::vector<double> generateRandomIndividual() {
        std::vector<double> individual(lb.size());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        for (size_t i = 0; i < lb.size(); ++i) {
            individual[i] = lb[i] + dist(rng) * (ub[i] - lb[i]);
        }
        
        return individual;
    }
    
     std::vector<std::vector<double>> optimize(
        std::function<double(const std::vector<double>&)> fitness_func) {
        
        // Generate initial population
        std::vector<std::vector<double>> population;
        for (int i = 0; i < population_size; ++i) {
            population.push_back(generateRandomIndividual());
        }
        
        for (int generation = 0; generation < max_generations; ++generation) {
            std::cout << "**********************************" << std::endl;
            std::cout << "Generation number : " << generation + 1 << std::endl;
            
            // Evaluate fitness
            std::vector<std::pair<double, int>> fitness_indices;
            for (int i = 0; i < population_size; ++i) {
                std::cout << "String : " << i + 1 << ", Gen : " << generation + 1 << std::endl;
                double fitness = fitness_func(population[i]);
                fitness_indices.push_back({fitness, i});
            }
            
            // Sort by fitness (assuming minimization)
            std::sort(fitness_indices.begin(), fitness_indices.end());
            
            std::cout << "Best cost for generation " << generation + 1 << " : " << fitness_indices[0].first << std::endl;
            
            // Check for convergence
            if (fitness_indices[0].first < tolerance) {
                break;
            }
            
            // Select parents
            std::vector<std::vector<double>> parents;
            for (int i = 0; i < num_parents; ++i) {
                parents.push_back(population[fitness_indices[i].second]);
            }
            
            // Generate children through crossover
            std::vector<std::vector<double>> children;
            std::uniform_real_distribution<double> alpha_dist(0.0, 1.0);
            
            for (int i = 0; i < num_children; i += 2) {
                if (i + 1 < num_children) {
                    double alpha = alpha_dist(rng);
                    double beta = alpha_dist(rng);
                    
                    std::vector<double> child1(lb.size()), child2(lb.size());
                    for (size_t j = 0; j < lb.size(); ++j) {
                        child1[j] = parents[i % num_parents][j] * alpha + 
                                   parents[(i + 1) % num_parents][j] * (1 - alpha);
                        child2[j] = parents[i % num_parents][j] * beta + 
                                   parents[(i + 1) % num_parents][j] * (1 - beta);
                        
                        // Ensure bounds
                        child1[j] = std::max(lb[j], std::min(ub[j], child1[j]));
                        child2[j] = std::max(lb[j], std::min(ub[j], child2[j]));
                    }
                    children.push_back(child1);
                    children.push_back(child2);
                }
            }
            
            // Create new population
            population.clear();
            population.insert(population.end(), parents.begin(), parents.end());
            population.insert(population.end(), children.begin(), children.end());
            
            // Fill remaining with random individuals
            while (population.size() < population_size) {
                population.push_back(generateRandomIndividual());
            }
        }
        
        return population;
    }
};

std::vector<double> bruteForceOptimization(
    std::vector<double>& var, 
    double desHoverTime, 
    bool constant_angles, 
    double T, 
    std::vector<double>& w, 
    double TOLNEO, 
    double TOLEarth, 
    double dT, 
    int NumSeg,
    std::vector<std::shared_ptr<OrbitingObject>>& bodies,
    int max_iter = 700, 
    int tolerance = 150, 
    double m = 0.2, 
    double momentum = 0.9, 
    int printStatements = 1) {
    
    double cost = lightSailCost(var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, NumSeg, dT, bodies);
    int no_change = 0;
    std::vector<double> velocity(var.size(), 0.0);

    for (int j = 0; j < max_iter; ++j) {
        for (int i = 1; i < static_cast<int>(var.size()); ++i) {
            if (no_change > tolerance) {
                m *= 0.5;
                no_change = 0;
            }

            // Create incremented variable copy
            std::vector<double> inc_var = var;
            inc_var[i] += inc_var[i] * m;
            
            // Create decremented variable copy
            std::vector<double> dec_var = var;
            dec_var[i] -= dec_var[i] * m;

            double inc_var_cost = lightSailCost(inc_var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, NumSeg, dT, bodies);
            double dec_var_cost = lightSailCost(dec_var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, NumSeg, dT, bodies);

            if (cost > inc_var_cost) {
                var = inc_var;
                cost = inc_var_cost;
                no_change = 0;
            } else {
                no_change += 1;
            }

            if (cost > dec_var_cost) {
                var = dec_var;
                cost = dec_var_cost;
                no_change = 0;
            } else {
                no_change += 1;
            }
            
            if (printStatements == 1) {
                std::cout << "Iter #: " << j << " var #: " << i << " m: " << m << " new cost: " << cost << std::endl;
            }
        }

        if (cost < 0) {
            break;
        }
    }

    return var;
}


// Add this function after the existing functions, before the main function
void saveTrajectoriesToFile(const std::vector<std::shared_ptr<OrbitingObject>>& bodies, 
                           double dT, const std::string& filename = "celestial_trajectories.txt") {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing!" << std::endl;
        return;
    }
    
    // Set precision for output
    file << std::fixed << std::setprecision(12);
    
    // Write header
    file << "# Celestial Body Trajectories" << std::endl;
    file << "# Time step (dT): " << dT << " days" << std::endl;
    file << "# Units: Positions in AU, Velocities in AU/s" << std::endl;
    file << "# Format: Time(days) Body_Name Pos_X Pos_Y Pos_Z Vel_X Vel_Y Vel_Z" << std::endl;
    file << "#" << std::endl;
    
    // Find the maximum number of data points
    size_t max_points = 0;
    for (const auto& body : bodies) {
        max_points = std::max(max_points, body->positions.size());
    }
    
    // Write data for each time step
    for (size_t i = 0; i < max_points; ++i) {
        double time_days = i * dT;
        
        for (const auto& body : bodies) {
            if (i < body->positions.size()) {
                Vector3D pos = body->positions[i];
                Vector3D vel = (i < body->velocities.size()) ? body->velocities[i] : Vector3D(0, 0, 0);
                
                file << time_days << " " 
                     << body->name << " "
                     << pos.x << " " << pos.y << " " << pos.z << " "
                     << vel.x << " " << vel.y << " " << vel.z << std::endl;
            }
        }
    }
    
    file.close();
    std::cout << "Trajectories saved to " << filename << std::endl;
    std::cout << "Total time points: " << max_points << std::endl;
    std::cout << "Total simulation time: " << (max_points - 1) * dT << " days" << std::endl;
}

// Alternative function to save each body to separate files
void saveTrajectoriesToSeparateFiles(const std::vector<std::shared_ptr<OrbitingObject>>& bodies, 
                                    double dT, const std::string& prefix = "trajectory_") {
    for (const auto& body : bodies) {
        std::string filename = prefix + body->name + ".txt";
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing!" << std::endl;
            continue;
        }
        
        // Set precision for output
        file << std::fixed << std::setprecision(12);
        
        // Write header
        file << "# Trajectory for " << body->name << std::endl;
        file << "# Time step (dT): " << dT << " days" << std::endl;
        file << "# Units: Positions in AU, Velocities in AU/s" << std::endl;
        file << "# Orbital elements: a=" << body->a << " AU, e=" << body->e 
             << ", i=" << body->i * 180/PI << " deg" << std::endl;
        file << "# Format: Time(days) Pos_X Pos_Y Pos_Z Vel_X Vel_Y Vel_Z" << std::endl;
        file << "#" << std::endl;
        
        // Write data
        for (size_t i = 0; i < body->positions.size(); ++i) {
            double time_days = i * dT;
            Vector3D pos = body->positions[i];
            Vector3D vel = (i < body->velocities.size()) ? body->velocities[i] : Vector3D(0, 0, 0);
            
            file << time_days << " "
                 << pos.x << " " << pos.y << " " << pos.z << " "
                 << vel.x << " " << vel.y << " " << vel.z << std::endl;
        }
        
        file.close();
        std::cout << "Trajectory for " << body->name << " saved to " << filename << std::endl;
        std::cout << "Data points: " << body->positions.size() << std::endl;
    }
}


int main() {
    try {
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;

        // Initialize Earth
        auto Earth = std::make_shared<OrbitingObject>(
            1.496430050492096e11/AU, 0.01644732672533337, 
            0.002948905108822335 * PI/180, 250.3338397589344 * PI/180,
            211.4594045093653 * PI/180, 188.288909341482 * PI/180,
            5.97219e24, "Earth"
        );
        
        // Initialize Bennu
        auto Bennu = std::make_shared<OrbitingObject>(
            1.684403508572353E+11 / AU, 2.037483028559170E-01,
            6.032932274441114E+00 * PI/180, 6.637388139157433E+01 * PI/180,
            1.981305199928344E+00 * PI/180, 2.175036361198920E+02 * PI/180,
            7.329E10, "Bennu"
        );
        
        double dT = 0.1;  // days 
        double T = 365 * 5;  // simulation time
        double desHoverTime = 80;  // days
        
        std::cout << "Calculating Earth trajectory..." << std::endl;
        calcCelestialTraj(*Earth, dT, T);
        
        std::cout << "Calculating Bennu trajectory..." << std::endl;
        calcCelestialTraj(*Bennu, dT, T);
        
        std::vector<std::shared_ptr<OrbitingObject>> bodies = {Earth, Bennu};
        saveTrajectoriesToFile(bodies, dT, "celestial_trajectories.txt");
        saveTrajectoriesToSeparateFiles(bodies, dT, "trajectory_");

        // Set up optimization parameters
        int NumSeg = 55;
        int SegLength = 55;
        std::vector<double> lb, ub;
        
        // Bounds setup
        lb.push_back(1);    // degree_min
        lb.push_back(150);  // time_min
        lb.push_back(30*1000/AU);  // vel_min
        
        ub.push_back(5);    // degree_max
        ub.push_back(365);  // time_max
        ub.push_back(30*1000/AU);  // vel_max
        
        // Add segment bounds (time segments)
        for (int i = 0; i < NumSeg; ++i) {
            lb.push_back(1);
            ub.push_back(SegLength);
        }
        
        // Add cone angle bounds
        for (int i = 0; i < NumSeg; ++i) {
            lb.push_back(-1.2);
            ub.push_back(1.2);
        }
        
        // Add clock angle bounds
        for (int i = 0; i < NumSeg; ++i) {
            lb.push_back(-6.2831/2);
            ub.push_back(6.2831/2);
        }
        
        // Weights
        std::vector<double> w = {1, 10, 10, 1, 1, 0};
        double TOLNEO = 1000/AU;
        double TOLEarth = 0.1;
        
        // Define fitness function
        auto fitness_func = [&](const std::vector<double>& vars) -> double {
            return lightSailCost(vars, desHoverTime, false, T, w, TOLNEO, TOLEarth, NumSeg, dT, bodies);
        };
        
        std::vector<double> best_var;
        
        // Genetic Algorithm
        int S = 30;   // population size
        int P = 10;   // number of parents
        int K = 10;   // number of children
        int G = 100;  // max generations
        double TOL = 1e-3;  // tolerance

        unsigned int fixed_seed = 12345; 
        GeneticAlgorithm ga(S, P, K, G, TOL, lb, ub, fixed_seed);
        
        std::cout << "Starting genetic algorithm optimization..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        auto result = ga.optimize(fitness_func);
        best_var = result[0];

        std::cout << "***************************************************************" << std::endl;
        std::cout << "Genetic algorithm optimization complete!" << std::endl;
        std::cout << "***************************************************************" << std::endl;
        // printDesignVariables(best_var, NumSeg);

        // Brute Force Optimization
        best_var = bruteForceOptimization(best_var, desHoverTime, false, T, w, TOLNEO, TOLEarth, dT, NumSeg, bodies);

        end_time = std::chrono::high_resolution_clock::now();
        std::cout << "***************************************************************" << std::endl;
        std::cout << "Brute force optimization complete!" << std::endl;
        std::cout << "***************************************************************" << std::endl;
        printDesignVariables(best_var, NumSeg);
        std::cout << "***************************************************************" << std::endl;
        
        std::cout << "The corresponding costs are :" << std::endl;
        
        // Calculate and print final cost with detailed breakdown
        LightSailSolver finalSolver(best_var, bodies);
        finalSolver.runSim(desHoverTime, false, T, TOLNEO, TOLEarth, NumSeg, dT);
        double finalCost = finalSolver.calcCost(w, TOLNEO, TOLEarth, true);  // true for print statements

        // Save the final sail trajectory
        std::cout << "Saving sail trajectory..." << std::endl;
        saveSailTrajectoryToFile(finalSolver, "final_sail_trajectory.txt");
        
        std::cout << "The minimum cost is: " << finalCost << std::endl;
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout<<"Total execution time : " << duration.count() / 1000000.0 <<" seconds." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}