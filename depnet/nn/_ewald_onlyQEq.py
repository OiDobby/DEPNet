import math
from typing import List, Optional

import numpy as np
import torch
from ase.data import covalent_radii
from e3nn.o3 import Irreps, Linear
from scipy import constants

from depnet.datasets import ELECTROSTATIC_ENERGY_KEY, TOTAL_CHARGE_KEY, STRESS_KEY, ELECTRODE_KEY, CHARGES_KEY, AREA_KEY
#from nequip.data import AtomicDataDict
from depnet.data import AtomicDataDict
from depnet.nequip_nn import GraphModuleMixin, ConvNetLayer, AtomwiseLinear

import time
import psutil

class Ewald(GraphModuleMixin, torch.nn.Module):
    """
    Layer to calculate electrostatic energy via Ewald summation
    """

    def __init__(
        self,
        allowed_species: List[int],  # automatically passed from shared_params
        out_field: str = ELECTROSTATIC_ENERGY_KEY,
        irreps_in=None,
        scale: float = 1.0,  # std of dataset, physical unit
    ):
        super().__init__()

        self.scale = scale

        self.out_field = out_field
        irreps_out = {self.out_field: Irreps("1x0e")}
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.POSITIONS_KEY, CHARGES_KEY],
            irreps_out=irreps_out,
        )

        # sigma: species_index (0-indexed) -> covalent radius
        self.sigma = torch.from_numpy(np.array(covalent_radii[allowed_species]))
        if len(allowed_species) == 1:
            self.sigma = torch.unsqueeze(self.sigma, 0)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        device = data[AtomicDataDict.POSITIONS_KEY].device
        species_idx = data[AtomicDataDict.SPECIES_INDEX_KEY]
        sigmas = self.sigma[species_idx].to(device)
        charges = data[CHARGES_KEY]  # (num_atoms, 1)

        ptr = data["ptr"]
        batch_size = ptr.shape[0] - 1

        ele = torch.zeros(batch_size, device=device)
        for bi in range(batch_size):
            cell_bi = data[AtomicDataDict.CELL_KEY][bi]
            pos_bi = data[AtomicDataDict.POSITIONS_KEY][ptr[bi] : ptr[bi + 1]]
            sigmas_bi = sigmas[ptr[bi] : ptr[bi + 1]]
            ewald_bi = EwaldAuxiliary(
                cell=cell_bi,
                pos=pos_bi,
                sigmas=sigmas_bi,
                point_charge=False,
            )

            # EwaldAuxiliary.calc_energy takes only 1-dimensional tensor
            charges_bi = torch.squeeze(charges[ptr[bi] : ptr[bi + 1]], dim=1)
            ele[bi] = ewald_bi.calc_energy(charges_bi) / self.scale

        data[self.out_field] = torch.unsqueeze(ele, dim=1)  # (batch_size, 1)

        return data


class EwaldQeq(GraphModuleMixin, torch.nn.Module):
    """
    Qeq for periodic systems
    """

    def __init__(
        self,
        allowed_species: List[int],  # automatically passed from shared_params
        out_field: str = ELECTROSTATIC_ENERGY_KEY, 
        irreps_in=None,
        scale: float = 1.0,  # std of dataset, physical unit
        use_dipole: bool = False,
        use_bg: bool = False,
        use_slab: bool = False,
        use_electrodeU: bool = False,
    ):
        super().__init__()

        self.scale = scale

        self.out_field = out_field
        ### additional block ###
        self.charges_key = "charges"
        self.area_key = "area"
        self.use_dipole = use_dipole
        self.use_bg = use_bg
        self.use_slab = use_slab
        self.use_electrodeU = use_electrodeU
        ########################
        irreps_out = {
            self.out_field: Irreps("1x0e"),
            self.charges_key:Irreps("1x0e"),
            self.area_key: Irreps("1x0e"),
            #CHARGES_KEY: Irreps("1x0e"),
        }

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.POSITIONS_KEY, AtomicDataDict.NODE_FEATURES_KEY],
            irreps_out=irreps_out,
        )

        # sigma: species_index (0-indexed) -> covalent radius
        self.sigma = torch.from_numpy(np.array(covalent_radii[allowed_species]))
        if len(allowed_species) == 1:
            self.sigma = torch.unsqueeze(self.sigma, 0)

        self.to_chi = Linear(
            irreps_in=irreps_in[AtomicDataDict.NODE_FEATURES_KEY],  
            irreps_out=Irreps("1x0e"),
        )

        hardness_params = torch.ones(len(allowed_species))
        self.to_hardness = torch.nn.Parameter(data=hardness_params)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        device = data[AtomicDataDict.POSITIONS_KEY].device
        ### original code lines ###
        #species_idx = data[AtomicDataDict.SPECIES_INDEX_KEY]
        #sigmas = self.sigma[species_idx].to(device)
        ### edited lines for matching device ###
        species_idx = data[AtomicDataDict.SPECIES_INDEX_KEY].to(device)
        self.sigma = self.sigma.to(device)
        sigmas = self.sigma[species_idx]
        chi = self.to_chi(data[AtomicDataDict.NODE_FEATURES_KEY])  # (num_atoms, 1)
        
        # square here to restrit hardness to be positive!
        hardness = torch.square(self.to_hardness[species_idx])  # (num_atoms, )

        ptr = data["ptr"]
        batch_size = ptr.shape[0] - 1
        '''
        # ===== edited block: Const. P
        if self.use_electrodeU:
            elec_U = data.get(AtomicDataDict.ELECTRODE_KEY, None)  # (batch_size,)
            if elec_U is None:
                self.use_electrodeU = False
        else:
            elec_U = None
        # =====
        '''

        ele = torch.zeros(batch_size, device=device)
        charges = []
        lambdas = []
        area_list = []

        for bi in range(batch_size):
            cell_bi = data[AtomicDataDict.CELL_KEY][bi]
            pos_bi = data[AtomicDataDict.POSITIONS_KEY][ptr[bi] : ptr[bi + 1]]
            sigmas_bi = sigmas[ptr[bi] : ptr[bi + 1]]
            ewald_bi = EwaldAuxiliary(
                cell=cell_bi,
                pos=pos_bi,
                sigmas=sigmas_bi,
                point_charge=False,
                use_slab=self.use_slab,
            )

            # coefficient matrix for Qeq
            hardness_bi = hardness[ptr[bi] : ptr[bi + 1]]
            num_atoms_bi = (ptr[bi + 1] - ptr[bi])
            coeffs_bi = torch.ones((int(num_atoms_bi + 1), int(num_atoms_bi + 1)), device=device)
            energy_matrix_bi = ewald_bi.get_qeq_matrix(hardness_bi)

            # ===== correction terms
            if self.use_dipole:
                energy_matrix_bi += Corrections.calc_dipole_matrix(cell=cell_bi, pos=pos_bi)

            if self.use_bg:
                energy_matrix_bi += Corrections.calc_bg_effect(cell=cell_bi, eta=ewald_bi.eta, num_atoms=num_atoms_bi, device=pos_bi.device)
            # =====

            coeffs_bi[:num_atoms_bi, :num_atoms_bi] = energy_matrix_bi
            coeffs_bi[-1, -1] = 0.0

            ### ori code line ###
            #total_charge_bi = torch.Tensor([[data[TOTAL_CHARGE_KEY][bi]]]).to(device)  # (1, 1)

            ### edited code lines - always zero ###
            total_charge_bi = torch.tensor([[0.0]]).to(device)  # (1, 1) always zero
            #######################################

            chi_bi = chi[ptr[bi] : ptr[bi + 1]]
            '''
            # ===== edited block: Const. P
            if self.use_electrodeU:
                chi_bi = chi_bi - elec_U[bi].expand(num_atoms_bi, 1)  # (num_atoms_bi, 1)
            # =====
            '''
            rhs_bi = torch.cat([-chi_bi, total_charge_bi])

            # solve Qeq
            # for small (n, n)-matrix (n < 2048), batched DGESV is faster than usual DGESV in MAGMA
            charges_and_lambda = torch.linalg.solve(
                torch.unsqueeze(coeffs_bi, dim=0), torch.unsqueeze(rhs_bi, dim=0)
            )
            charges_bi = torch.squeeze(charges_and_lambda, dim=0)[:-1]  # (num_atoms_bi, 1)
            charges.append(charges_bi)

            # ===== additional block
            charges_bi = charges_bi.squeeze()

            # Extract the last element as lambda
            lambda_bi = torch.squeeze(charges_and_lambda, dim=0)[-1]  # Get the last element
            lambda_bi = lambda_bi.view(1)
            lambdas.append(lambda_bi)
            # =====

            # Compute total charge after calculating charges_bi
            total_charge_bi = torch.sum(charges_bi)

            # minimized electrostatic energy
            e_qeq_bi = 0.5 * torch.sum(
                energy_matrix_bi * charges_bi[:, None] * charges_bi[None, :]
            )
            e_qeq_bi = e_qeq_bi + torch.sum(charges_bi * chi_bi)

            ele[bi] = e_qeq_bi / self.scale

            area_list.append(ewald_bi.area)

        #data[CHARGES_KEY] = torch.cat(charges)  # (num_atoms, 1)
        data[self.charges_key] = torch.cat(charges) ### edited line
        data[self.out_field] = torch.unsqueeze(ele, dim=1)  # (batch_size, 1)

        area_tensor = torch.stack(area_list)
        data[self.area_key] = area_tensor.unsqueeze(1)

        return data

class EwaldAuxiliary:
    """
    Parameters
    ----------
    cell: (3, 3)
        cell[i] is the i-th lattice vector
    pos: (num_atoms, 3)
    sigmas: (num_atoms, 1)
        sigmas[i] is the width of the gaussian of the i-th atom
        if point_charge=True, sigmas=None is permitted.
    eta: width of screening gaussian
    cutoff_real: cutoff radius for real part
    cutoff_recip: cutoff radius for reciprocal part
    point_charge: iff true, width of gaussian charges `sigmas` are ignored
    eps: small epsilon to avoid zero division
    """

    # ke = e^{2}/(4 * pi * epsilon_{0}) = 2.3070775523417355e-28 J.m = 14.399645478425668 eV.ang
    #COULOMB_FACTOR = 1e10 * constants.e / (4 * math.pi * constants.epsilon_0)
    #COULOMB_FACTOR = 14.399645478425668  # eV.ang

    def __init__(
        self,
        cell: torch.Tensor,
        pos: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        ### ori code lines ###
        #eta: Optional[float] = None,
        #cutoff_real: Optional[float] = None,
        #cutoff_recip: Optional[float] = None,
        ### edited code lines ###
        eta: Optional[torch.Tensor] = None,
        cutoff_real: Optional[torch.Tensor] = None,
        cutoff_recip: Optional[torch.Tensor] = None,
        #########################
        accuracy: float = 1e-5,
        point_charge: bool = True,
        eps: float = 1e-8,
        use_slab: bool = False,
    ):
        #self.cell = cell
        #self.volume = torch.abs(torch.dot(self.cell[0], torch.cross(self.cell[1], self.cell[2])))
        #self.pos = pos
        #self.sigmas = sigmas

        ### additional code lines ###
        self.COULOMB_FACTOR = 14.399645478425668  # eV.ang

        self.device = pos.device  # Initialize self.device from the position tensor
        self.cell = cell.to(self.device)
        self.volume = torch.abs(torch.dot(self.cell[0], torch.cross(self.cell[1], self.cell[2], dim=0)))
        self.area = torch.linalg.norm(torch.cross(self.cell[0], self.cell[1], dim=0))
        self.pos = pos
        self.use_slab = use_slab

        if sigmas is None:
            raise ValueError("sigmas can not be set None. It should be Tensor.")

        self.sigmas = sigmas.to(pos.device)  # sigmas move to pos's device
        ##############################

        self.eta = (
            #eta
            #if eta
            eta if eta is not None
            #else ((self.volume ** 2 / self.pos.shape[0]) ** (1 / 6)) / math.sqrt(2.0 * math.pi)
            else ((self.volume ** 2 / self.pos.shape[0]) ** (1 / 6)) / math.sqrt(math.pi)
        )
        # ===== eta for slab system
        '''
        self.eta = torch.tensor(0.0)
        if eta is not None:
            self.eta = eta
        else:
            eta_3D = ((self.volume ** 2 / self.pos.shape[0]) ** (1 / 6)) / math.sqrt(math.pi)
            eta_2D = ((self.area ** 2 / self.pos.shape[0]) ** (1 / 4)) / math.sqrt(math.pi)
            self.eta = eta_2D if self.use_slab else eta_3D
        '''
        # =====

        self.cutoff_real = (
            #cutoff_real if cutoff_real else math.sqrt(-2.0 * math.log(accuracy)) * self.eta
            cutoff_real if cutoff_real is not None else math.sqrt(-math.log(2.0 * accuracy)) * self.eta
            #cutoff_real if cutoff_real is not None else math.sqrt(-math.log(2.0 * accuracy)) * (self.eta * math.sqrt(2.0))
        )
        self.cutoff_recip = (
            #cutoff_recip if cutoff_recip else math.sqrt(-2.0 * math.log(accuracy)) / self.eta
            cutoff_recip if cutoff_recip is not None else math.sqrt(-math.log(2.0 * accuracy)) / self.eta
            #cutoff_recip if cutoff_recip is not None else math.sqrt(-math.log(2.0 * accuracy)) / (self.eta * math.sqrt(2.0))
        )

        self.point_charge = point_charge
        self.eps = eps

        # precompute energy matrices
        # ===== full matrix (original)
        #e_real_matrix = self._calc_real_energy_matrix()
        #e_recip_matrix = self._calc_reciprocal_energy_matrix()
        # ===== vectorization and upper off-diagonal
        displacements = self.get_disp_matrix()
        if self.use_slab:
            e_real, filtered_indices_real = self._calc_real_energy_vector_slab(displacements)
            e_recip, filtered_indices_recip = self._calc_reciprocal_energy_vector_slab(displacements)
        else:
            e_real, filtered_indices_real = self._calc_real_energy_vector(displacements)
            e_recip, filtered_indices_recip = self._calc_reciprocal_energy_vector(displacements)

        e_merge = torch.cat([e_real, e_recip], dim=0)
        filtered_indices = torch.cat([filtered_indices_real, filtered_indices_recip], dim=0)  # (num_pairs, 2)

        num_atoms = self.num_atoms
        e_total_matrix = torch.zeros((num_atoms, num_atoms), device=self.pos.device)
        # === 1st version
        #e_total_matrix[filtered_indices[:, 0], filtered_indices[:, 1]] += e_merge
        #e_total_matrix[filtered_indices[:, 1], filtered_indices[:, 0]] = e_total_matrix[filtered_indices[:, 0], filtered_indices[:, 1]]
        # === 2nd version (best)
        e_merge = e_merge.to(e_total_matrix.dtype)
        e_total_matrix.scatter_add_(0, filtered_indices[:, 0].unsqueeze(1), e_merge.unsqueeze(1))
        e_total_matrix.scatter_add_(0, filtered_indices[:, 1].unsqueeze(1), e_merge.unsqueeze(1))
        # =====
        e_self_matrix = self._calc_self_energy_matrix()
        e_total_matrix = e_total_matrix + e_self_matrix
        self._e_total_matrix = e_total_matrix
        #self._e_total_matrix = e_real_matrix + e_recip_matrix + e_self_matrix

    @property
    def num_atoms(self):
        return self.pos.shape[0]

    @property
    def energy_matrix(self):
        """
        total energy matrix e_{ij}
        Ewald summation is obtained by `0.5 * sum_{i,j} e_{ij} q_{i} q_{j}`
        """
        return self._e_total_matrix

    def get_qeq_matrix(self, hardness):
        """
        return energy matrix with hardness term

        Parameters
        ----------
        hardness: (num_atoms, 1)
        """
        ### ori code lines ###
        #assert hardness.shape[0] == self.num_atoms
        #mat = torch.clone(self.energy_matrix)
        #mat += torch.diag(torch.squeeze(hardness))
        ### edit code lines ###
        if len(hardness.shape) == 2 and hardness.shape[1] == 1:
            hardness = torch.squeeze(hardness)  # (num_atoms,)
        assert hardness.shape[0] == self.num_atoms, \
            f"Expected hardness of shape ({self.num_atoms},), but got {hardness.shape}"
        mat = torch.clone(self.energy_matrix)
        mat = mat + torch.diag(hardness)
        #######################
        return mat

    def calc_energy(self, charges: torch.Tensor):
        """
        Calculate electrostatic energy by Ewald summation

        Parameters
        ----------
        charges: (num_atoms, 1)
        """
        e_total = 0.5 * torch.sum(self.energy_matrix * charges[:, None] * charges[None, :])
        return e_total

    def _calc_real_energy_matrix(self):
        """
        Calculate real-space-part energy in atomic unit
        """
        ### additional code line ###
        if self.sigmas is None or not isinstance(self.sigmas, torch.Tensor):
            raise ValueError("The type of sigmas is must be torch.Tensor.")
        ############################
        # calculate length between atoms `i` and `j` with `shift`
        shifts = get_shifts_within_cutoff(self.cell, self.cutoff_real)  # (num_shifts, 3)
        # disps_ij[i, j, :] is displacement vector r_{ij}
        disps_ij = self.pos[None, :, :] - self.pos[:, None, :]
        disps = disps_ij[None, :, :, :] + torch.matmul(shifts, self.cell)[:, None, None, :]
        distances_all = torch.linalg.norm(disps, dim=-1)  # (num_shifts, num_atoms, num_atoms)

        # retrieve pairs whose length are shorter than cutoff
        within_cutoff = (distances_all > self.eps) & (distances_all < self.cutoff_real)
        distances = distances_all[within_cutoff]

        e_real_matrix_aug = torch.zeros_like(distances_all)
        e_real_matrix_aug[within_cutoff] = torch.erfc(distances / (math.sqrt(2) * self.eta))

        if not self.point_charge:
            gammas_all = torch.sqrt(
                torch.square(self.sigmas[:, None]) + torch.square(self.sigmas[None, :])
            )
            gammas = torch.broadcast_to(gammas_all, distances_all.shape)[within_cutoff]
            e_real_matrix_aug[within_cutoff] -= torch.erfc(distances / (math.sqrt(2) * gammas))
        e_real_matrix_aug[within_cutoff] /= distances
        e_real_matrix = self.COULOMB_FACTOR * torch.sum(
            e_real_matrix_aug, dim=0
        )  # sum over shifts
        return e_real_matrix

    def _calc_reciprocal_energy_matrix(self):
        # calculate reciprocal points
        recip = get_reciprocal_vectors(self.cell)
        shifts = get_shifts_within_cutoff(recip, self.cutoff_recip)  # (num_shifts, 3)
        ks_all = torch.matmul(shifts, recip)
        length_all = torch.linalg.norm(ks_all, dim=-1)  # (num_shifts, )

        # retrieve reciprocal points whose length are shorter than cutoff
        within_cutoff = (length_all > self.eps) & (length_all < self.cutoff_recip)
        ks = ks_all[within_cutoff]
        length = length_all[within_cutoff]
        # disps_ij[i, j, :] is displacement vector r_{ij}, (num_atoms, num_atoms, 3)
        disps_ij = self.pos[None, :, :] - self.pos[:, None, :]
        phases = torch.sum(ks[:, None, None, :] * disps_ij[None, :, :, :], dim=-1)

        e_recip_matrix_aug = (
            torch.cos(phases)
            * torch.exp(-0.5 * torch.square(self.eta * length[:, None, None]))
            / torch.square(length[:, None, None])
        )
        e_recip_matrix = (
            self.COULOMB_FACTOR
            * 4.0
            * math.pi
            / self.volume
            * torch.sum(e_recip_matrix_aug, dim=0)
        )
        return e_recip_matrix

    def _calc_self_energy_matrix(self):
        device = self.pos.device
        #diag = -math.sqrt(2.0 / math.pi) / self.eta * torch.ones(self.num_atoms, device=device)  # original code line; wrong mathematics
        #diag = -1.0 / (math.sqrt(2.0 * math.pi) * self.eta) * torch.ones(self.num_atoms, device=device)  # correct mathematics
        diag = -1.0 / (math.sqrt(math.pi) * self.eta) * torch.ones(self.num_atoms, device=device)  # eta variation
        if not self.point_charge:
            diag = diag + (1.0 / (math.sqrt(math.pi) * self.sigmas))
        e_self_matrix = self.COULOMB_FACTOR * torch.diag(diag)
        return e_self_matrix

    def get_disp_matrix(self):
        """
        Calculate displacement matrix using a nested loop to save memory.
        Args:
            positions: (num_atoms, 3) Atomic positions.
        Returns:
            displacements: Tensor of shape (num_pairs, 5), where each row is (i, j, dx, dy, dz).
        """
        # ===== broadcating
        num_atoms = self.pos.shape[0]

        # Broadcasting to compute displacement matrix
        pos_expanded = self.pos.unsqueeze(0)  # (1, num_atoms, 3)
        disp_matrix = pos_expanded - pos_expanded.transpose(0, 1)  # (num_atoms, num_atoms, 3)

        # Mask upper triangular part (ensure dtype=torch.bool)
        upper_tri_mask = torch.triu(torch.ones(num_atoms, num_atoms, device=self.pos.device, dtype=torch.bool), diagonal=1)
        i_indices, j_indices = torch.where(upper_tri_mask)
        displacements_vectors = disp_matrix[upper_tri_mask]

        # Combine indices and displacements
        displacements = torch.empty((i_indices.shape[0], 5), device=self.pos.device)
        displacements[:, 0] = i_indices
        displacements[:, 1] = j_indices
        displacements[:, 2:] = displacements_vectors

        return displacements

    def _calc_real_energy_vector(self, displacements):
        """
        Calculate Real-space energy using displacement matrix with loops.
        Args:
            displacements: List of tuples (i, j, displacement).
        Returns:
            e_real_matrix: Real-space energy matrix.
        """
        # Convert displacements to Tensor
        indices = displacements[:, :2].long()  # (num_pairs, 2)
        disp_vectors = displacements[:, 2:]    # (num_pairs, 3)

        # Generate shifts
        shifts = get_shifts_within_cutoff(self.cell, self.cutoff_real)
        shift_vectors = torch.matmul(shifts, self.cell)  # (M, 3)

        # Broadcasting shifts to all displacement vectors
        shifted_displacements = disp_vectors.unsqueeze(1) + shift_vectors.unsqueeze(0)  # (num_pairs, M, 3)
        distances = torch.linalg.norm(shifted_displacements, dim=-1)  # (num_pairs, M)

        # Apply cutoff filtering
        within_cutoff = (distances > self.eps) & (distances < self.cutoff_real)
        filtered_distances = distances[within_cutoff]
        filtered_indices = indices.unsqueeze(1).expand(-1, len(shifts), -1)[within_cutoff]

        # Compute energy
        #e_real_terms = torch.erfc(filtered_distances / (math.sqrt(2) * self.eta)) / filtered_distances
        e_real_terms = torch.erfc(filtered_distances / (self.eta)) / filtered_distances

        # Gaussian charge correction if point_charge is False
        if not self.point_charge:
            i_indices, j_indices = filtered_indices[:, 0], filtered_indices[:, 1]
            gammas = torch.sqrt(self.sigmas[i_indices] ** 2 + self.sigmas[j_indices] ** 2)  # Combined sigmas
            e_real_terms = e_real_terms - (torch.erfc(filtered_distances / (math.sqrt(2) * gammas)) / filtered_distances)
        
        e_real_terms = e_real_terms * self.COULOMB_FACTOR
        return e_real_terms, filtered_indices
    
    def _calc_reciprocal_energy_vector(self, displacements):
        """
        Calculate Reciprocal-space energy using displacement matrix.
        Args:
            displacements: List of tuples (i, j, displacement).
        Returns:
            e_recip_matrix: Reciprocal-space energy matrix.
        """
        # Calculate reciprocal lattice vectors and shifts
        recip = get_reciprocal_vectors(self.cell)
        shifts = get_shifts_within_cutoff(recip, self.cutoff_recip)
        shift_vectors = torch.matmul(shifts, recip)
        k_norms = torch.linalg.norm(shift_vectors, dim=-1)
        
        # Apply cutoff filtering
        valid_mask = (k_norms > self.eps) & (k_norms < self.cutoff_recip)
        shift_vectors = shift_vectors[valid_mask]
        k_norms = k_norms[valid_mask]
        gaussian_decay = torch.exp(-0.25 * (self.eta * k_norms) ** 2) / (k_norms ** 2 + self.eps)

        # Extract displacement components
        indices = displacements[:, :2].long()
        disp_vectors = displacements[:, 2:]

        # Compute phases and energy terms
        phases = torch.matmul(disp_vectors.unsqueeze(1), shift_vectors.T)
        e_recip_terms = torch.sum(torch.cos(phases) * gaussian_decay, dim=-1).flatten()

        e_recip_terms = e_recip_terms * (self.COULOMB_FACTOR * 4.0 * math.pi / self.volume)
        return e_recip_terms, indices

    def _calc_real_energy_vector_slab(self, displacements):
        """
        Calculate Real-space energy in 2.5D (slab structure) using displacement matrix.
        Args:
            displacements: Tensor of shape (num_pairs, 5), where each row is (i, j, dx, dy, dz).
        Returns:
            e_real_terms: Tensor of real-space energy terms.
            filtered_indices: Tensor of indices corresponding to valid interactions.
        """
        indices = displacements[:, :2].long()  # Atom pair indices
        disp_vectors = displacements[:, 2:]  # (num_pairs, 3) -> (dx, dy, dz)
        #distances = torch.linalg.norm(disp_vectors, dim=-1)  # 3D distances (dx, dy, dz)

        # Generate shifts for only x and y
        shifts = get_shifts_within_cutoff_2d(self.cell, self.cutoff_real)  # 2D shifts only
        shift_vectors = torch.matmul(shifts, self.cell)  # (M, 2)

        # Apply 2D shifts to x and y components only
        shifted_displacements = disp_vectors.unsqueeze(1)  # (num_pairs, 1, 3)
        shifted_displacements = shifted_displacements + shift_vectors.unsqueeze(0)

        shifted_distances = torch.linalg.norm(shifted_displacements, dim=-1)

        # Apply cutoff filtering
        within_cutoff = (shifted_distances > self.eps) & (shifted_distances < self.cutoff_real)
        filtered_distances = shifted_distances[within_cutoff]
        filtered_indices = indices.unsqueeze(1).expand(-1, shifts.shape[0], -1)[within_cutoff]

        # Compute energy
        e_real_terms = torch.erfc(filtered_distances / self.eta) / filtered_distances

        # Gaussian charge correction if point_charge is False
        if not self.point_charge:
            i_indices, j_indices = filtered_indices[:, 0], filtered_indices[:, 1]
            gammas = torch.sqrt(self.sigmas[i_indices] ** 2 + self.sigmas[j_indices] ** 2)
            e_real_terms = e_real_terms - (torch.erfc(filtered_distances / (math.sqrt(2) * gammas)) / filtered_distances)
            
        e_real_terms = e_real_terms * self.COULOMB_FACTOR
        return e_real_terms, filtered_indices

    def _calc_reciprocal_energy_vector_slab(self, displacements):
        """
        Calculate Reciprocal-space energy with z-direction interaction but no periodicity,
        explicitly distinguishing between k=0 and k≠0 contributions.
        """
        # Compute reciprocal lattice vectors for x, y only
        recip = get_reciprocal_vectors(self.cell)  # Full 3D reciprocal lattice (3, 3)
        shifts = get_shifts_within_cutoff_2d(recip, self.cutoff_recip)  # Compute 2D shift vectors (N, 3), z=0
        shift_vectors = torch.zeros((shifts.size(0), 3), device=shifts.device)  # Initialize 3D shifts
        shift_vectors[:, :2] = torch.matmul(shifts, recip[:, :2])  # Apply 2D shifts to x, y components

        # Compute norms of reciprocal vectors
        k_norms = torch.linalg.norm(shift_vectors, dim=-1)  # (N,)

        # **Extract displacement components**
        indices = displacements[:, :2].long()  # (num_pairs, 2)
        disp_vectors = displacements[:, 2:]    # (num_pairs, 3)
        z_ij = disp_vectors[:, 2]              # (num_pair,)

        # **Compute k=0 Contribution (Long-Range Correction) Only for k=0 Pairs**
        erf_term = torch.erf(z_ij / self.eta)
        exp_term = torch.exp(-z_ij**2 / self.eta**2) / math.sqrt(math.pi)
        e_recip_k_zero = (z_ij * erf_term + self.eta * exp_term)
        e_recip_k_zero *= -self.COULOMB_FACTOR * (2 * math.pi / self.area)

        # **Compute Reciprocal Energy for k ≠ 0 with cutoff**
        valid_mask = (k_norms > self.eps) & (k_norms < self.cutoff_recip)  # filtering
        shift_vectors = shift_vectors[valid_mask]  # (num_k, 3)
        k_norms = k_norms[valid_mask]              # (num_k,)

        z_ij_exp = z_ij[:, None]                # (num_pairs, 1)
        k_norms_exp = k_norms[None, :]          # (1, num_k)

        # Phase terms
        phases = torch.matmul(disp_vectors.unsqueeze(1), shift_vectors.T).squeeze(1)  # (num_pairs, num_k)

        # Damping correction terms
        exp_kz_erfc1 = torch.exp(k_norms_exp * z_ij_exp) * torch.erfc(z_ij_exp / self.eta + (k_norms_exp * self.eta / 2))
        exp_kz_erfc2 = torch.exp(-k_norms_exp * z_ij_exp) * torch.erfc(-z_ij_exp / self.eta + (k_norms_exp * self.eta / 2))
        correction_factor = exp_kz_erfc1 + exp_kz_erfc2  # (num_pairs, num_k)

        e_recip_k_nonzero = (torch.cos(phases) * correction_factor / k_norms_exp).sum(dim=-1)
        e_recip_k_nonzero *= self.COULOMB_FACTOR * (math.pi / self.area)

        # **Final Reciprocal Energy including k=0 and k≠0 Contributions**
        e_recip_terms = e_recip_k_nonzero + e_recip_k_zero  # (num_pairs,)
        return e_recip_terms, indices

def get_reciprocal_vectors(cell):
    """
    Return reciprocal vectors of `cell`.
    Let the returned matrix be recip, dot(cell[i, :], recip[j, :]) = 2 * pi * (i == j)
    """
    recip = 2 * math.pi * torch.transpose(torch.linalg.inv(cell), 0, 1)
    return recip


def get_shifts_within_cutoff(cell, cutoff):
    """
    Return all shifts required to search for atoms within cutoff
    """
    device = cell.device

    # projected length for three planes
    proj = torch.zeros(3, device=device)
    nx = torch.cross(cell[1], cell[2])
    ny = torch.cross(cell[2], cell[0])
    nz = torch.cross(cell[0], cell[1])
    proj[0] = torch.dot(cell[0], nx / torch.linalg.norm(nx))
    proj[1] = torch.dot(cell[1], ny / torch.linalg.norm(ny))
    proj[2] = torch.dot(cell[2], nz / torch.linalg.norm(nz))

    shift = torch.ceil(cutoff / torch.abs(proj))

    ### ori code lines ###
    #grid = torch.cartesian_prod(
    #    torch.arange(-shift[0], shift[0] + 1, device=device),
    #    torch.arange(-shift[1], shift[1] + 1, device=device),
    #    torch.arange(-shift[2], shift[2] + 1, device=device),
    #)
    ######################

    grids = torch.cartesian_prod(
        torch.arange(-shift[0].item(), shift[0].item() + 1, device=device),
        torch.arange(-shift[1].item(), shift[1].item() + 1, device=device),
        torch.arange(-shift[2].item(), shift[2].item() + 1, device=device),
    )

    # Filter shifts that are outside the cutoff radius
    grid = grids[
        torch.norm(torch.matmul(grids.float(), cell), dim=1) <= cutoff
    ]

    return grid

def get_reciprocal_vectors_2d(cell):
    """
    Return reciprocal vectors for 2D `cell`.
    Let the returned matrix be recip, dot(cell[i, :], recip[j, :]) = 2 * pi * (i == j)
    Args:
        cell: (2, 2) Tensor representing the 2D lattice.
    Returns:
        recip: (2, 2) Tensor of reciprocal lattice vectors.
    """
    device = cell.device

    # Extract the 2D lattice from the 3D cell
    cell_2d = cell[:2, :2]  # Use only x-y components of the cell

    # Calculate the 2D reciprocal lattice vectors
    recip_2d = 2 * math.pi * torch.transpose(torch.linalg.inv(cell_2d), 0, 1)

    # Expand to 3D by adding zeros in the z-direction
    recip = torch.zeros(3, 3, device=device)
    recip[:2, :2] = recip_2d  # Add x, y reciprocal lattice components

    return recip

def get_shifts_within_cutoff_2d(cell, cutoff):
    """
    Return all shifts required to search for atoms within cutoff in 2D.
    Args:
        cell: (3, 3) Tensor, lattice vectors including z-direction.
        cutoff: Scalar, cutoff distance for neighbor search.
    Returns:
        grid: (N, 3) Tensor of shifts, where z-direction is always 0.
    """
    device = cell.device

    # Extract 2D lattice (x, y components only)
    cell_2d = cell[:2, :2]  # Use the top-left 2x2 part of the 3x3 cell

    # Projected length for x and y planes
    proj = torch.zeros(2, device=device, dtype=torch.float32)
    z_hat = torch.tensor([0.0, 0.0, 1.0], device=device)
    nx = torch.cross(
        torch.stack([cell_2d[1, 0], cell_2d[1, 1], torch.tensor(0.0, device=device)], dim=0),
        z_hat
    )
    ny = torch.cross(
        torch.stack([cell_2d[0, 0], cell_2d[0, 1], torch.tensor(0.0, device=device)], dim=0),
        z_hat
    )

    proj[0] = torch.dot(
        torch.stack([cell_2d[0, 0], cell_2d[0, 1], torch.tensor(0.0, device=device)], dim=0),
        nx / torch.linalg.norm(nx),
    )
    proj[1] = torch.dot(
        torch.stack([cell_2d[1, 0], cell_2d[1, 1], torch.tensor(0.0, device=device)], dim=0),
        ny / torch.linalg.norm(ny),
    )

    # Calculate shift range for x and y only
    shift_xy = torch.ceil(cutoff / torch.abs(proj))

    # Generate 2D grid (z-direction fixed to 0)
    grids = torch.cartesian_prod(
        torch.arange(-shift_xy[0].item(), shift_xy[0].item() + 1, device=device),
        torch.arange(-shift_xy[1].item(), shift_xy[1].item() + 1, device=device),
    )
    grids = torch.cat([grids, torch.zeros(grids.size(0), 1, device=device)], dim=1)

    # Filter shifts within cutoff distance (only x-y plane considered)
    grid = grids[
        torch.norm(torch.matmul(grids[:, :2].float(), cell[:2, :2]), dim=1) <= cutoff
    ]

    return grid

class Corrections:
    @staticmethod
    def calc_dipole_matrix(cell: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Calculate the dipole correction matrix using vectorized operations.

        Args:
            cell (torch.Tensor): Lattice vectors (3, 3).
            pos (torch.Tensor): Atom positions (num_atoms, 3).

        Returns:
            torch.Tensor: Dipole correction matrix (num_atoms, num_atoms).
        """
        volume = torch.abs(torch.dot(cell[0], torch.cross(cell[1], cell[2], dim=0)))
        lz = torch.norm(cell[2])  # Length of the z-direction
        z_wrapped = pos[:, 2] % cell[2, 2]
        num_atoms = pos.shape[0]

        # Mask upper triangular part
        upper_tri_mask = torch.triu(torch.ones(num_atoms, num_atoms, device=pos.device, dtype=torch.bool), diagonal=1)
        i_indices, j_indices = torch.where(upper_tri_mask)
        z_i = z_wrapped[i_indices]
        z_j = z_wrapped[j_indices]

        # Compute off-diagonal terms
        off_diagonal_terms = (4 * math.pi / volume) * (z_i * z_j)

        # Compute diagonal terms
        diagonal_terms = (2 * math.pi / volume) * z_wrapped**2 - (math.pi * lz**2) / (3 * volume)

        # Initialize full matrix
        dipole_matrix = torch.zeros((num_atoms, num_atoms), device=pos.device)

        # Fill upper triangular and diagonal values
        dipole_matrix[i_indices, j_indices] = off_diagonal_terms
        dipole_matrix.diagonal().copy_(diagonal_terms)

        # Symmetrize the matrix
        return 2 * (dipole_matrix + dipole_matrix.T)

    @staticmethod
    def calc_bg_effect(cell: torch.Tensor, eta: float, num_atoms: int, device: torch.device) -> torch.Tensor:
        """
        Calculate the background charge correction matrix using vectorized operations.

        Args:
            cell (torch.Tensor): Lattice vectors (3, 3).
            eta (float): Width of the screening Gaussian.
            num_atoms (int): Number of atoms in the system.
            device (torch.device): Device for tensor operations.

        Returns:
            torch.Tensor: Background charge correction matrix (num_atoms, num_atoms).
        """
        volume = torch.abs(torch.dot(cell[0], torch.cross(cell[1], cell[2], dim=0)))
        bg_effect = -math.pi * (eta**2) / volume
        return 2 * bg_effect * torch.ones((num_atoms, num_atoms), device=device)
