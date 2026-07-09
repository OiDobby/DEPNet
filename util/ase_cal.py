import json
import sys
import numpy as np
import torch
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.data import vdw_radii
from ase.calculators import calculator
from time import time

# Input block
device = 'cpu'  # Set device ('cuda' for GPU, 'cpu' for CPU)
deployed_model_path = "./deploy_model_SC.pth"  # Path to deployed NequIP model
input_file = '../temp.extxyz'  # Path to input file in extxyz format
outfile_path = 'nequip_output.txt'  # Path for saving output results

# Setting up PyTorch configurations
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True

# Importing NequIP calculator
from nequip.dynamics.nequip_calculator import NequIPCalculator
from ase import Atoms
from nequip.data import AtomicData

# Initializing NequIP calculator with deployed model
calculator = NequIPCalculator.from_deployed_model(
    model_path=deployed_model_path,
    species_to_type_name={"Sr": "Sr", "Ti": "Ti", "O": "O"},
    device=device
)

# Load atomic positions from input file
atom_pos = read(input_file, format='extxyz')
atom_pos.calc = calculator

# Start time measurement
start_time = time()

# Perform calculations
energy = atom_pos.get_potential_energy()
forces = atom_pos.get_forces()
stress = atom_pos.get_stress() * 1602.1766208  # Stress conversion factor

# Retrieve charges if available in calculator results
charges_key = getattr(calculator, "charges_key", "charges")  # Default key is "charges"
charges = calculator.results.get(charges_key, None)

# Calculate total charge if charges are present
total_charge = None
if charges is not None:
    total_charge = np.sum(charges)

# End time measurement
end_time = time()
execution_time = end_time - start_time

# Save results to output file
with open(outfile_path, 'w') as f:
    f.write(f"Energy: {energy}\n")
    f.write("Forces:\n")
    for force in forces:
        f.write(f"[{force[0]}, {force[1]}, {force[2]}]\n")
    
    # Write stress as a single line
    f.write("Stresses (kbar):\n")
    f.write(f"[{', '.join(map(str, stress.flatten()))}]\n")

    # Write charges if available
    if charges is not None:
        f.write("Charges:\n")
        for charge in charges:
            f.write(f"{charge}\n")
        f.write(f"Total Charge: {total_charge}\n")
    
    # Write execution time in seconds
    f.write(f"Execution Time (seconds): {execution_time:.2f}\n")

print(f"Data saved to {outfile_path}")
