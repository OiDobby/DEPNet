class Correction:
    """
    Class to handle correction terms for Ewald-based calculations:
    - Dipole Correction
    - Background Charge Correction
    - Gaussian Charge Correction

    This class provides methods to compute various correction terms
    that account for periodic boundary conditions (PBC) and charge distribution effects.
    """

    COULOMB_FACTOR = 14.399645478425668  # eV·Å (Coulomb constant for energy conversion)

    def __init__(self):
        pass

    def get_dipole_corr(self, charges, positions, cell):
        """
        Compute the dipole correction term for periodic boundary conditions.

        Parameters:
        - charges: Tensor of charges (num_atoms,).
        - positions: Tensor of atomic positions (num_atoms, 3).
        - cell: Tensor representing the unit cell (3x3) in Å.

        Returns:
        - Dipole correction energy (eV).
        
        The correction accounts for the z-directional dipole moment and other factors
        related to the total charge and atomic positions within the simulation cell.
        """
        # Calculate the volume of the simulation cell
        volume = torch.abs(torch.dot(cell[0], torch.cross(cell[1], cell[2])))

        # Wrap z-coordinates to ensure positions are within the simulation cell
        z_wrapped = positions[:, 2] % cell[2, 2]

        # Compute the z-direction dipole moment
        mz = torch.sum(charges * z_wrapped)

        # Compute the total charge
        total_charge = torch.sum(charges)

        # Compute Σ(q_i * z_i^2)
        q_z_squared = torch.sum(charges * z_wrapped**2)

        # Compute the z-direction cell length (Lz)
        lz = torch.norm(cell[2])

        # Combine terms to compute the dipole correction energy
        e_corr_dipole = self.COULOMB_FACTOR * (2 * math.pi / volume) * (
            mz**2 - total_charge * q_z_squared - total_charge**2 * (lz**2 / 12)
        )
        return e_corr_dipole

    def get_bg_corr(self, total_charge, cell, kappa):
        """
        Compute the background charge correction term.

        Parameters:
        - total_charge: Scalar representing the total charge in the system.
        - cell: Tensor representing the unit cell (3x3) in Å.
        - kappa: Ewald splitting parameter (1/Å).

        Returns:
        - Background charge correction energy (eV).
        
        The correction adjusts the energy for a uniform background charge that neutralizes the total system charge.
        """
        # Calculate the volume of the simulation cell
        volume = torch.abs(torch.dot(cell[0], torch.cross(cell[1], cell[2])))

        # Compute the background charge correction energy
        return - (math.pi / 2) * (self.COULOMB_FACTOR * total_charge**2) / (volume * kappa**2)

    def get_gaussian_corr(self, charges, positions, sigmas, cell, cutoff):
        """
        Compute the Gaussian charge correction term.

        Parameters:
        - charges: Tensor of charges (num_atoms,).
        - positions: Tensor of atomic positions (num_atoms, 3).
        - sigmas: Tensor of Gaussian widths (num_atoms,).
        - cell: Tensor representing the unit cell (3x3) in Å.
        - cutoff: Real-space cutoff distance for neighbor interaction (Å).

        Returns:
        - Gaussian charge correction energy (eV).
        
        The correction adjusts the energy for interactions between Gaussian-distributed charges
        and includes a self-energy term for each charge.
        """
        # Use the existing function to get all shifts within the cutoff
        shifts = get_shifts_within_cutoff(cell, cutoff)

        # Compute pairwise displacements with periodic boundary conditions
        disps = positions[None, :, :] - positions[:, None, :]  # (N, N, 3)
        disps_all = disps[None, :, :, :] + torch.matmul(shifts, cell)[:, None, None, :]  # (num_shifts, N, N, 3)

        # Compute distances between atoms
        distances_all = torch.linalg.norm(disps_all, dim=-1)  # (num_shifts, N, N)

        # Filter distances within the cutoff and exclude self-interactions
        within_cutoff = (distances_all > 1e-8) & (distances_all < cutoff)
        distances = distances_all[within_cutoff]

        # Compute gamma_ij = sqrt(sigma_i^2 + sigma_j^2) for each pair
        sigmas_i = sigmas[:, None]  # Broadcast sigma_i
        sigmas_j = sigmas[None, :]  # Broadcast sigma_j
        gammas_all = torch.sqrt(sigmas_i**2 + sigmas_j**2)  # (N, N)
        gammas = gammas_all[within_cutoff]  # Select gammas within cutoff

        # Compute interaction term
        interaction_term = -0.5 * torch.sum(
            (charges[:, None] * charges[None, :])[within_cutoff]
            * torch.erfc(distances / (math.sqrt(2) * gammas))
            / distances
        )

        # Compute self-energy term
        self_energy_term = torch.sum(charges**2 / (2 * math.sqrt(math.pi) * sigmas))

        # Combine terms to compute the Gaussian correction energy
        e_corr_gauss = self.COULOMB_FACTOR * (interaction_term + self_energy_term)

        return e_corr_gauss

