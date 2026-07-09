import time
import torch
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from ase.filters import ExpCellFilter, FrechetCellFilter
from nequip.dynamics.nequip_calculator import NequIPCalculator

# === User-defined inputs ===
device = 'cpu'  # Set to 'cuda' for GPU usage if available, 'cpu' can be available
deployed_model_path = "./deploy_model_SC.pth"  # Path to the NequIP model file
input_structure_path = '../temp.extxyz'  # Path to the structure file for the simulation
output_relaxed_structure_path = 'structure_relax.extxyz'  # Path to save relaxed structure
step_output_path = "relax_steps.log"  # Path for each step's output
total_output_path = "relax_total.log"  # Path for logging final result and total time
species_to_type_name = {"Sr": "Sr", "Ti": "Ti", "O": "O"}  # Element and model name mapping
fmax = 0.01  # Force convergence criterion in eV/Å
optimizer = "LBFGS" # Default to "LBFGS", user can set to "BFGS" if desired
filter_choice = "FrechetCellFilter"  # Default filter; can also be set to "ExpCellFilter"
# ===========================

# Set up the calculator model
calculator = NequIPCalculator.from_deployed_model(
    model_path=deployed_model_path,
    species_to_type_name=species_to_type_name,
    device=device
)

# Load the structure and assign the calculator
atoms = read(input_structure_path, format='extxyz')
atoms.calc = calculator

# Choose cell filter based on user input
if filter_choice == "ExpCellFilter":
    relaxation_filter = ExpCellFilter(atoms)
else:
    relaxation_filter = FrechetCellFilter(atoms)  # Default to FrechetCellFilter

# Choose optimizer based on user input
if optimizer == "BFGS":
    opt = BFGS(relaxation_filter, logfile=total_output_path)
else:
    opt = LBFGS(relaxation_filter, logfile=total_output_path)

# Custom callback to log energy, forces, stress, charges, and time per step
step_times = []
start_time = time.time()

def log_step_details():
    step_start_time = time.time()

    # Compute energy, forces, stress, and charges
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress() * 1602.1766208  # Convert stress to kBar
    charges_key = getattr(calculator, "charges_key", "charges")
    charges = calculator.results.get(charges_key, None)
    total_charge = np.sum(charges) if charges is not None else None

    # Calculate step duration
    step_duration = time.time() - step_start_time
    step_times.append(step_duration)

    # Log details to the step output file
    with open(step_output_path, "a") as f:
        f.write(f"Step {len(step_times)}:\n")
        f.write(f"Energy = {energy:.4f} eV\n")
        f.write("Forces = \n")
        f.write(f"{forces}\n")
        f.write("Stress (kBar) =\n")
        f.write(f"[{' '.join(f'{s:.6f}' for s in stress)}]\n")

        if charges is not None:
            f.write("Charges = \n")
            f.write(f"{charges}\n")
            f.write(f"Total Charge = {total_charge}\n")

        f.write(f"Calculation time (seconds): {step_duration:.4f}\n\n")

    print(f"Step {len(step_times)} complete. Results saved to {step_output_path}")

# Attach the logging function to the optimizer to run after each step
opt.attach(log_step_details)

# Run the relaxation until forces are below the fmax threshold
opt.run(fmax=fmax)

# Save the relaxed structure
write(output_relaxed_structure_path, atoms)
print(f"Relaxed structure saved to {output_relaxed_structure_path}")

# Calculate and log the total relaxation time
total_relaxation_time = time.time() - start_time
with open(step_output_path, "a") as f:
    f.write(f"Total relaxation time (seconds): {total_relaxation_time:.4f}\n")
with open(total_output_path, "a") as log_file:
    log_file.write(f"Total relaxation time (seconds): {total_relaxation_time:.4f}\n")

print(f"Total relaxation time: {total_relaxation_time:.4f} seconds")
