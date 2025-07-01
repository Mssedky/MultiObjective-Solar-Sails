import src_python_c2.loadVars as ld
import src_python_c2.body_data as bd
import run_optimizer as op
from src_python_c2.solver import *
from src_python_c2.test_helpers import *


if __name__ == "__main__":

    AU = 1.496e11 # Astronomical Unit (m)

    dT, T, numSails, NumSeg, SegLength, constant_angles, w = op.optimizer_input_data()

    # Choose filename
    filename = f"export_{numSails}_sails_data.txt"
    
    # Save sail trajectory data
    save_traj = 1

    # Load variables
    varSol = ld.load_design_variables('Case 2/output/solar_exit_design_variables_GA_5_Sails.txt')

    # Load bodies
    bodies = bd.get_body_data(dT,T)

    dv=3+NumSeg*3

    run_test(lightSailSolver, constant_angles, T, NumSeg, dT, numSails, bodies, dv, varSol, w, filename, save_traj)
    


   


