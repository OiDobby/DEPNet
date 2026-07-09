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

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True

#device = 'cuda'
device = 'cpu'
from nequip.dynamics.nequip_calculator import NequIPCalculator
from ase import Atoms
from nequip.data import AtomicData

# Deploy model path
deployed_model_path = "./deploy_model_SC.pth"
calculator = NequIPCalculator.from_deployed_model(model_path=deployed_model_path,
                                             species_to_type_name={"Au": "Au",
                                                                   "Mg": "Mg",
                                                                   "O": "O" ,
                                                                   "Al": "Al"},
                                             device=device)

atom_pos = read('../temp.extxyz', format='extxyz')

atom_pos.calc = calculator

energy = atom_pos.get_potential_energy()
forces = atom_pos.get_forces()
stress = atom_pos.get_stress()

charges_key = getattr(calculator, "charges_key", "charges")  # "charges" set to default value
charges = calculator.results.get(charges_key, None)

total_charge = None
if charges is not None:
    total_charge = np.sum(charges)

#print(f"energy: {energy}")
#print(f"Forces: {forces}")

outfile_path = 'nequip_output.txt'
# Save to txt
with open(outfile_path, 'w') as f:
    f.write(f"Energy: {energy}\n")
    f.write("Forces:\n")
    for force in forces:
        f.write(f"[{force[0]}, {force[1]}, {force[2]}]\n")
    f.write("Stresses:")
    if stress.ndim == 2 and stress.shape == (3, 3):  # check (3, 3) type
            for row in stress:
                f.write(f"[{', '.join(map(str, row))}]\n")  # each row converts to string
    else:
        f.write(f"{stress}\n")

    if charges is not None:
        f.write("Charges:\n")
        for charge in charges:
            f.write(f"{charge}\n")
        f.write(f"Total Charge: {total_charge}\n")

print(f"Data saved to {outfile_path}")
