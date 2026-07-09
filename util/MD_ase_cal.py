import time
import torch
import numpy as np
from ase import units
from ase import Atoms
from ase.md.langevin import Langevin
from ase.io import read
from nequip.dynamics.nequip_calculator import NequIPCalculator
from nequip.data import AtomicDataDict

# === User-defined inputs ===
device = 'cpu'  # Set to 'cuda' to use GPU if available
deployed_model_path = "./deploy_model_SC.pth"  # Path to the NequIP model file
input_structure_path = '../temp.extxyz'  # Path to the structure file for the simulation
log_file_path = "md_simulation_output.log"  # Path to the output log file

# MD simulation parameters
temperature = 300  # in Kelvin
time_step = 1 * units.fs  # Time interval in femtoseconds
total_steps = 100  # Total number of simulation steps

species_to_type_name = {"Sr": "Sr", "Ti": "Ti", "O": "O"}  # Element and model name mapping
# ===========================

# Set up the calculator model
calculator = NequIPCalculator.from_deployed_model(
    model_path=deployed_model_path,
    species_to_type_name=species_to_type_name,
    device=device
)

# Create ASE Atoms object and assign the calculator
atom_pos = read(input_structure_path, format='extxyz')
atom_pos.calc = calculator

# Set up MD simulation
dyn = Langevin(atom_pos, time_step, temperature * units.kB, friction=0.02)

# Start the overall timer
overall_start_time = time.time()

# MD simulation loop
with open(log_file_path, "w") as log_file:
    for step in range(total_steps):
        start_time = time.time()
        dyn.run(1)  # Execute 1 step

        # Calculate energy, forces, stress, and charges
        energy = atom_pos.get_potential_energy()
        forces = atom_pos.get_forces()
        stress = atom_pos.get_stress() * 1602.1766208  # Convert to kBar

        # Reshape stress to a single line for the log
        stress_formatted = ' '.join(f"{s:.6f}" for s in stress)

        charges_key = getattr(calculator, "charges_key", "charges")  # Default to "charges"
        charges = calculator.results.get(charges_key, None)

        # Calculate total charge
        total_charge = np.sum(charges) if charges is not None else None

        # Step completion time
        total_time = time.time() - start_time

        # Write results to log file
        log_file.write(f"Step {step + 1}:\n")
        log_file.write(f"Energy = {energy:.4f} eV\n")
        log_file.write("Forces = \n")
        log_file.write(f"{forces}\n")
        log_file.write("Stress (kBar) =\n")
        log_file.write(f"[{stress_formatted}]\n")
        
        if charges is not None:
            log_file.write("Charges = \n")
            log_file.write(f"{charges}\n")
            log_file.write(f"Total Charge = {total_charge}\n")

        log_file.write(f"Calculation time (sec): {total_time:.4f}\n\n")

        # Console output (optional)
        print(f"Step {step + 1} complete. Results saved to {log_file_path}")

    # Calculate total time for all steps
    overall_total_time = time.time() - overall_start_time
    log_file.write(f"Total calculation time (sec): {overall_total_time:.4f}\n")

print("Simulation complete. Total results saved to", log_file_path)
