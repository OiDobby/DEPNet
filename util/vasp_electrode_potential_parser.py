#!/usr/bin/env python3
"""
Parse electrode-potential targets from VASP OUTCAR files and convert
the corresponding structures to an extxyz dataset for DEPNet.

Supported VACUUM_LEVEL_MODE values
----------------------------------
"delta"
    Store the difference between the upper- and lower-side vacuum levels:

        electrode_potential = V_upper - V_lower

    This is the mode used for the final DEPNet training dataset.

"upper"
    Use the upper-side vacuum level and store:

        electrode_potential = E_F - V_upper

"lower"
    Use the lower-side vacuum level and store:

        electrode_potential = E_F - V_lower

"avg"
    Use the average vacuum level and store:

        electrode_potential = E_F - (V_upper + V_lower) / 2

Optional features
-----------------
- Bader or Hirshfeld charge parsing
- AMIX-based filtering

Only the atomic structure and the configuration-level
"electrode_potential" property are required for DEPNet training.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ase.io
import numpy as np
from ase import Atoms


# =============================================================================
# User settings
# =============================================================================

BASE_DIRECTORY = "./data"
OUTPUT_EXTXYZ = "dataset.extxyz"
ERROR_LOG_FILE = "error_log.txt"

# Available modes: "delta", "avg", "upper", "lower"
VACUUM_LEVEL_MODE = "delta"

# Keep "run_" if calculations are stored in run_* directories.
# Set to None to process every directory containing an OUTCAR.
RUN_PREFIX: Optional[str] = "run_"

# Optional charge parsing
USE_CHARGE = False
CHARGE_TYPE = "bader"  # "bader" or "hirshfeld"
BADER_FILENAME = "ACF.dat"
HIRSHFELD_FILENAME = "hirshfeld_block.dat"

# Optional AMIX filtering
USE_AMIX_FILTER = False
TARGET_AMIX = 0.4
AMIX_TOL = 1e-8


# =============================================================================
# Directory utilities
# =============================================================================

def collect_calculation_dirs(
    base_dir: str,
    run_prefix: Optional[str] = "run_",
) -> List[Path]:
    """Return calculation directories containing an OUTCAR."""
    base = Path(base_dir)
    calc_dirs: List[Path] = []

    for root, dirs, files in os.walk(base):
        dirs.sort()
        root_path = Path(root)

        if "OUTCAR" not in files:
            continue

        if run_prefix is None or root_path.name.startswith(run_prefix):
            calc_dirs.append(root_path)

    return sorted(calc_dirs, key=lambda p: str(p))


# =============================================================================
# VASP electrode-potential parsing
# =============================================================================

def parse_outcar_vacuum_and_fermi(
    outcar_file: Path,
    method: str = "delta",
) -> Tuple[float, float]:
    """
    Extract the vacuum level(s) and Fermi level from a VASP OUTCAR.

    Returns
    -------
    vacuum_value : float
        "upper" -> V_upper
        "lower" -> V_lower
        "avg"   -> (V_upper + V_lower) / 2
        "delta" -> V_upper - V_lower
    fermi_level : float
        The last E-fermi value found in the OUTCAR.
    """
    vacuum_upper = None
    vacuum_lower = None
    fermi_level = None

    marker = "vacuum level on the upper side and lower side of the slab"

    with outcar_file.open("r", errors="replace") as f:
        for line in f:
            if "E-fermi" in line:
                fermi_level = float(line.split()[2])

            if marker in line:
                columns = line.split()
                vacuum_upper = float(columns[-2])
                vacuum_lower = float(columns[-1])

    if fermi_level is None:
        raise ValueError(f"Fermi level not found in {outcar_file}.")

    if vacuum_upper is None or vacuum_lower is None:
        raise ValueError(
            f"Upper/lower vacuum levels not found in {outcar_file}. "
            "Check the slab calculation and dipole-correction settings."
        )

    if method == "upper":
        vacuum_value = vacuum_upper
    elif method == "lower":
        vacuum_value = vacuum_lower
    elif method == "avg":
        vacuum_value = 0.5 * (vacuum_upper + vacuum_lower)
    elif method == "delta":
        vacuum_value = vacuum_upper - vacuum_lower
    else:
        raise ValueError(
            f"Unknown VACUUM_LEVEL_MODE: {method}. "
            "Choose from 'delta', 'avg', 'upper', or 'lower'."
        )

    return vacuum_value, fermi_level


def get_electrode_potential(
    outcar_file: Path,
    method: str = "delta",
) -> float:
    """
    Return the value stored as atoms.info["electrode_potential"].

    Definitions
    -----------
    delta:
        V_upper - V_lower

    upper/lower/avg:
        -(V_vac - E_F) = E_F - V_vac
    """
    vacuum_value, fermi_level = parse_outcar_vacuum_and_fermi(
        outcar_file,
        method=method,
    )

    if method == "delta":
        return float(vacuum_value)

    return float(-(vacuum_value - fermi_level))


def parse_outcar_amix(outcar_file: Path) -> float:
    """Extract AMIX from a VASP OUTCAR."""
    with outcar_file.open("r", errors="replace") as f:
        for line in f:
            match = re.search(r"\bAMIX\s*=\s*([-+0-9Ee.]+)", line)
            if match:
                return float(match.group(1))

    raise ValueError(f"AMIX not found in {outcar_file}.")


# =============================================================================
# Optional charge parsing
# =============================================================================

def parse_acf(file_path: Path) -> List[float]:
    """Parse Bader electron populations from ACF.dat."""
    populations: List[float] = []
    lines = file_path.read_text(errors="replace").splitlines()

    start = None
    end = None

    for i, line in enumerate(lines):
        if "-----" in line and start is None:
            start = i + 1
        elif "-----" in line and start is not None:
            end = i
            break

    if start is None or end is None:
        raise ValueError(f"Could not identify the atomic block in {file_path}.")

    for line in lines[start:end]:
        columns = line.split()
        if len(columns) >= 5:
            populations.append(float(columns[4]))

    if not populations:
        raise ValueError(f"No Bader populations found in {file_path}.")

    return populations


def parse_hirshfeld_block(file_path: Path) -> List[float]:
    """Parse Hirshfeld electron populations from a critic2 output block."""
    populations: List[float] = []
    inside_block = False

    with file_path.open("r", errors="replace") as f:
        for line in f:
            if "* Integrated atomic properties" in line:
                inside_block = True
                continue

            if not inside_block:
                continue

            if (
                line.strip() == ""
                or line.startswith("----")
                or line.startswith("Sum")
            ):
                break

            if line.strip().startswith("#"):
                continue

            parts = line.strip().split()
            if len(parts) < 8:
                continue

            populations.append(float(parts[7]))

    if not populations:
        raise ValueError(f"No Hirshfeld populations found in {file_path}.")

    return populations


def parse_outcar_valence(outcar_file: Path) -> Dict[str, float]:
    """Extract element-wise ZVAL values from a VASP OUTCAR."""
    valence_dict: Dict[str, float] = {}
    element_list: List[str] = []
    zval_values: List[str] = []
    reading_zval = False

    with outcar_file.open("r", errors="replace") as f:
        for line in f:
            if "VRHFIN =" in line:
                parts = line.split("=")[1].strip().split(":")
                if len(parts) > 1:
                    element_list.append(parts[0].strip())

            if reading_zval:
                zval_values = (
                    line.replace("ZVAL", "")
                    .replace("=", "")
                    .strip()
                    .split()
                )
                break

            if "Ionic Valenz" in line:
                reading_zval = True

    if not zval_values:
        raise ValueError(f"Could not find ZVAL values in {outcar_file}.")

    if len(element_list) != len(zval_values):
        raise ValueError(
            "Mismatch between the number of elements and ZVAL values "
            f"in {outcar_file}."
        )

    for element, zval in zip(element_list, zval_values):
        valence_dict[element] = float(zval)

    return valence_dict


def populations_to_net_charges(
    populations: List[float],
    structure: Atoms,
    valence_dict: Dict[str, float],
) -> np.ndarray:
    """Convert electron populations to net charges: q = ZVAL - population."""
    if len(populations) != len(structure):
        raise ValueError(
            f"Charge count ({len(populations)}) does not match "
            f"the number of atoms ({len(structure)})."
        )

    charges = []

    for atom, population in zip(structure, populations):
        element = atom.symbol

        if element not in valence_dict:
            raise ValueError(f"Missing ZVAL for element {element}.")

        charges.append(valence_dict[element] - population)

    return np.asarray(charges, dtype=float)


def attach_optional_charges(
    structure: Atoms,
    calc_dir: Path,
    outcar_file: Path,
) -> None:
    """Attach per-atom charges when USE_CHARGE=True."""
    if CHARGE_TYPE == "bader":
        charge_file = calc_dir / BADER_FILENAME
        parser = parse_acf
    elif CHARGE_TYPE == "hirshfeld":
        charge_file = calc_dir / HIRSHFELD_FILENAME
        parser = parse_hirshfeld_block
    else:
        raise ValueError(
            f"Unknown CHARGE_TYPE: {CHARGE_TYPE}. "
            "Choose 'bader' or 'hirshfeld'."
        )

    if not charge_file.exists():
        raise FileNotFoundError(f"Missing charge file: {charge_file}")

    populations = parser(charge_file)
    valence_dict = parse_outcar_valence(outcar_file)
    charges = populations_to_net_charges(
        populations,
        structure,
        valence_dict,
    )

    structure.set_array("charges", charges)


# =============================================================================
# extxyz preparation
# =============================================================================

def make_clean_atoms(atoms: Atoms) -> Atoms:
    """
    Return a minimal ASE Atoms object suitable for extxyz output.

    DEPNet requires the structure and the configuration-level
    "electrode_potential" target. Charges are retained only when requested.
    """
    clean = Atoms(
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
    )

    clean.info["electrode_potential"] = float(
        atoms.info["electrode_potential"]
    )

    if "charges" in atoms.arrays:
        clean.set_array(
            "charges",
            np.asarray(atoms.arrays["charges"], dtype=float),
        )

    return clean


def process_calculation(calc_dir: Path) -> Atoms:
    """Read one VASP calculation and prepare one extxyz configuration."""
    outcar = calc_dir / "OUTCAR"

    structure = ase.io.read(
        str(outcar),
        format="vasp-out",
        index=-1,
    )

    structure.info["electrode_potential"] = get_electrode_potential(
        outcar,
        method=VACUUM_LEVEL_MODE,
    )

    if USE_CHARGE:
        attach_optional_charges(
            structure,
            calc_dir,
            outcar,
        )

    return make_clean_atoms(structure)


def collect_data_to_extxyz(
    base_dir: str,
    output_file: str,
    log_file_path: str,
) -> None:
    """Collect VASP calculations and write one extxyz dataset."""
    atoms_list: List[Atoms] = []
    skipped = 0

    calc_dirs = collect_calculation_dirs(
        base_dir,
        RUN_PREFIX,
    )

    with open(log_file_path, "w") as log_file:
        log_file.write("DEPNet VASP electrode-potential parsing log\n")
        log_file.write("=" * 50 + "\n")

        for calc_dir in calc_dirs:
            outcar = calc_dir / "OUTCAR"

            try:
                if USE_AMIX_FILTER:
                    amix = parse_outcar_amix(outcar)

                    if abs(amix - TARGET_AMIX) > AMIX_TOL:
                        log_file.write(
                            f"Skipped {calc_dir}: "
                            f"AMIX={amix} "
                            f"(expected {TARGET_AMIX})\n"
                        )
                        skipped += 1
                        continue

                atoms = process_calculation(calc_dir)
                atoms_list.append(atoms)

            except Exception as exc:
                log_file.write(f"Skipped {calc_dir}: {exc}\n")
                skipped += 1

    if not atoms_list:
        raise RuntimeError(
            "No valid structures were processed. "
            f"See {log_file_path}."
        )

    ase.io.write(
        output_file,
        atoms_list,
        format="extxyz",
    )

    print(f"Wrote {len(atoms_list)} configurations to {output_file}")
    print(
        f"Skipped {skipped} calculation(s). "
        f"See {log_file_path} for details."
    )


if __name__ == "__main__":
    collect_data_to_extxyz(
        BASE_DIRECTORY,
        OUTPUT_EXTXYZ,
        ERROR_LOG_FILE,
    )
