import numpy as np

def load_design_variables(filename):
    print(f"Loading design variables from {filename}...")
    with open(filename, 'r') as f:
        lines = f.readlines()
        # Concatenate all lines containing design variables
        design_variables_str = ''.join(lines[1:]).strip()
        # Convert the concatenated string to a numpy array
        design_variables = np.fromstring(design_variables_str.strip('[]'), sep=',')
    print("Done loading.")
    return design_variables