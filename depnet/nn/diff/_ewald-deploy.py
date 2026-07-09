import math
from typing import List, Optional

import numpy as np
import torch
from ase.data import covalent_radii
from e3nn.o3 import Irreps, Linear
from scipy import constants

from depnet.datasets import ELECTROSTATIC_ENERGY_KEY, TOTAL_CHARGE_KEY, STRESS_KEY ###CHARGES_KEY
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


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
    ):
        super().__init__()

        self.scale = scale

        self.out_field = out_field
        self.charges_key = "charges"
        irreps_out = {
            self.out_field: Irreps("1x0e"),
            self.charges_key:Irreps("1x0e"),
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
        #print("Keys in data:", list(data.keys()))
        species_idx = data[AtomicDataDict.SPECIES_INDEX_KEY]
        sigmas = self.sigma[species_idx].to(device)
        chi = self.to_chi(data[AtomicDataDict.NODE_FEATURES_KEY])  # (num_atoms, 1)
        # square here to restrit hardness to be positive!
        hardness = torch.square(self.to_hardness[species_idx])  # (num_atoms, )

        ptr = data["ptr"]
        batch_size = ptr.shape[0] - 1

        ele = torch.zeros(batch_size, device=device)
        charges = []
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

            # coefficient matrix for Qeq
            hardness_bi = hardness[ptr[bi] : ptr[bi + 1]]
            num_atoms_bi = (ptr[bi + 1] - ptr[bi])
            coeffs_bi = torch.ones((int(num_atoms_bi + 1), int(num_atoms_bi + 1)), device=device)
            energy_matrix_bi = ewald_bi.get_qeq_matrix(hardness_bi)
            coeffs_bi[:num_atoms_bi, :num_atoms_bi] = energy_matrix_bi
            coeffs_bi[-1, -1] = 0.0

            ### ori code line ###
            #total_charge_bi = torch.Tensor([[data[TOTAL_CHARGE_KEY][bi]]]).to(device)  # (1, 1)

            ### edited code lines 1 - tot_charge from reading ###
            #total_charge = data.get(TOTAL_CHARGE_KEY, 0.0)
            #if isinstance(total_charge, (list, np.ndarray, torch.Tensor)):
            #    total_charge = total_charge[0]
            #total_charge_bi = torch.tensor([[total_charge]]).to(device)  # (1, 1)
            #####################################################
            ### edited code lines 2 - always zero ###
            total_charge_bi = torch.tensor([[0.0]]).to(device)  # (1, 1) always zero
            #########################################

            chi_bi = chi[ptr[bi] : ptr[bi + 1]]
            rhs_bi = torch.cat([-chi_bi, total_charge_bi])

            # solve Qeq
            # for small (n, n)-matrix (n < 2048), batched DGESV is faster than usual DGESV in MAGMA
            charges_and_lambda = torch.linalg.solve(
                torch.unsqueeze(coeffs_bi, dim=0), torch.unsqueeze(rhs_bi, dim=0)
            )
            charges_bi = torch.squeeze(charges_and_lambda, dim=0)[:-1]  # (num_atoms_bi, 1)
            charges.append(charges_bi)

            # minimized electrostatic energy
            e_qeq_bi = 0.5 * torch.sum(
                energy_matrix_bi * charges_bi[:, None] * charges_bi[None, :]
            )
            e_qeq_bi += torch.sum(charges_bi * chi_bi)
            ele[bi] = e_qeq_bi / self.scale

        #data[CHARGES_KEY] = torch.cat(charges)  # (num_atoms, 1)
        data[self.charges_key] = torch.cat(charges) ### edited line
        data[self.out_field] = torch.unsqueeze(ele, dim=1)  # (batch_size, 1)

        #print("Keys in data:", list(data.keys()))

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
    COULOMB_FACTOR = 14.399645478425668  # eV.ang

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
        accuracy: float = 1e-8,
        point_charge: bool = True,
        eps: float = 1e-8,
    ):
        self.cell = cell
        self.volume = torch.abs(torch.dot(self.cell[0], torch.cross(self.cell[1], self.cell[2])))

        ### additional code lines ###
        if sigmas is None:
            raise ValueError("sigmas can not be set None. It should be Tensor.")
        self.sigmas = sigmas.to(pos.device)  # sigmas를 pos의 디바이스로 이동

        self.COULOMB_FACTOR = 14.399645478425668
        ##############################

        self.pos = pos
        self.sigmas = sigmas

        self.eta = (
            #eta
            #if eta
            eta if eta is not None
            else ((self.volume ** 2 / self.pos.shape[0]) ** (1 / 6)) / math.sqrt(2.0 * math.pi)
        )
        self.cutoff_real = (
            #cutoff_real if cutoff_real else math.sqrt(-2.0 * math.log(accuracy)) * self.eta
            cutoff_real if cutoff_real is not None else math.sqrt(-2.0 * math.log(accuracy)) * self.eta
        )
        self.cutoff_recip = (
            #cutoff_recip if cutoff_recip else math.sqrt(-2.0 * math.log(accuracy)) / self.eta
            cutoff_recip if cutoff_recip is not None else math.sqrt(-2.0 * math.log(accuracy)) / self.eta
        )

        self.point_charge = point_charge
        self.eps = eps

        # precompute energy matrices
        e_real_matrix = self._calc_real_energy_matrix()
        e_recip_matrix = self._calc_reciprocal_energy_matrix()
        e_self_matrix = self._calc_self_energy_matrix()
        self._e_total_matrix = e_real_matrix + e_recip_matrix + e_self_matrix

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
        assert hardness.shape[0] == self.num_atoms
        mat = torch.clone(self.energy_matrix)
        mat += torch.diag(torch.squeeze(hardness))
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
            raise ValueError("sigmas는 반드시 torch.Tensor이어야 합니다.")
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
        diag = -math.sqrt(2.0 / math.pi) / self.eta * torch.ones(self.num_atoms, device=device)
        if not self.point_charge:
            diag += 1.0 / (math.sqrt(math.pi) * self.sigmas)
        e_self_matrix = self.COULOMB_FACTOR * torch.diag(diag)
        return e_self_matrix


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

    #print("shift:", shift[0])
    #print("shift.itme():", shift[0].item())

    ### ori code lines ###
    #grid = torch.cartesian_prod(
    #    torch.arange(-shift[0], shift[0] + 1, device=device),
    #    torch.arange(-shift[1], shift[1] + 1, device=device),
    #    torch.arange(-shift[2], shift[2] + 1, device=device),
    #)
    ######################

    grid = torch.cartesian_prod(
        torch.arange(-shift[0].item(), shift[0].item() + 1, device=device),
        torch.arange(-shift[1].item(), shift[1].item() + 1, device=device),
        torch.arange(-shift[2].item(), shift[2].item() + 1, device=device),
    )

    return grid
