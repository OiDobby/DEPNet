from typing import Union
import torch
import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from depnet.data import AtomicData, AtomicDataDict
import depnet.scripts.nequip_deploy


class NequIPCalculator(Calculator):
    """NequIP ASE Calculator."""

    #implemented_properties = ["energy", "forces"] ### original code line
    implemented_properties = ["energy", "energies", "forces", "stress", "free_energy"] ### edit code line

    def __init__(
        self,
        model: torch.jit.ScriptModule,
        r_max: float,
        device: Union[str, torch.device],
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.model = model
        self.r_max = r_max
        self.device = device
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

    @classmethod
    def from_deployed_model(
        cls, model_path, device: Union[str, torch.device] = "cpu", **kwargs
    ):
        # load model
        model, metadata = depnet.scripts.nequip_deploy.load_deployed_model(
            model_path=model_path, device=device
        )
        model.eval()
        #r_max = float(metadata[depnet.scripts.nequip_deploy.R_MAX_KEY]) ### original code line
        # Use torch.no_grad if necessary
        with torch.no_grad():
            r_max = float(metadata[depnet.scripts.nequip_deploy.R_MAX_KEY])

        # build nequip calculator
        return cls(model=model, r_max=r_max, device=device, **kwargs)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Calculate properties.

        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
        data = data.to(self.device) ### original code line

        ### additional code lines ###
        '''
        data_dict = AtomicData.to_AtomicDataDict(data)

        if "ptr" not in data_dict:
            atomic_numbers = data_dict[AtomicDataDict.ATOMIC_NUMBERS_KEY]
            #data_dict["ptr"] = torch.tensor([0] + [len(atomic_numbers)])
            data_dict["ptr"] = torch.tensor(
                [0, int(atomic_numbers.shape[0])],
                dtype=torch.long,
                device=self.device,
            )

        data_dict = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data_dict.items()}
        '''
        # 1) AtomicData -> dict (once)
        ad = AtomicData.to_AtomicDataDict(data)
        
        # 2) tensor to device
        ad = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in ad.items()}

        # 3) ensuring single sample ptr
        N = int(ad[AtomicDataDict.ATOMIC_NUMBERS_KEY].shape[0])
        ad["ptr"] = torch.tensor([0, N], dtype=torch.long, device=self.device)

        # 4) (optional) standardization cell forming (3,3)
        def _ensure_cell_33(c, ref=None):
            if not torch.is_tensor(c):
                c = torch.as_tensor(np.asarray(c),
                                    dtype=(ref.dtype if torch.is_tensor(ref) else torch.float32),
                                    device=(ref.device if torch.is_tensor(ref) else "cpu"))
            if c.ndim == 1:
                if c.numel() == 9: return c.view(3,3)
                if c.numel() == 3: return torch.diag(c)
                raise RuntimeError("cell 1D must have 3 or 9 elements")
            if c.ndim == 2:
                if tuple(c.shape) == (3,3): return c
                if c.numel() == 9 and min(c.shape) == 1: return c.reshape(3,3)
                if c.numel() == 3 and min(c.shape) == 1: return torch.diag(c.reshape(3))
                raise RuntimeError("unsupported 2D cell")
            if c.ndim == 3 and c.shape[-2:] == (3,3): return c
            raise RuntimeError("unsupported cell rank")

        # 5) Batch dimensional ensuring: important!
        if AtomicDataDict.CELL_KEY in ad:
            ref  = ad.get(AtomicDataDict.POSITIONS_KEY, None)
            cell = _ensure_cell_33(ad[AtomicDataDict.CELL_KEY], ref=ref)  # -> (3,3) or (B,3,3)

            # (3,3) single sample -> make to (1,3,3) for adapting batch indexing
            if cell.ndim == 2:
                cell = cell.unsqueeze(0)         # xxx -> (1,3,3)

        ad[AtomicDataDict.CELL_KEY] = cell

        # (optional) pbc also adapted to batch
        if AtomicDataDict.PBC_KEY in ad:
            pbc = ad[AtomicDataDict.PBC_KEY]
            if torch.is_tensor(pbc) and pbc.ndim == 1 and pbc.numel() == 3:
                ad[AtomicDataDict.PBC_KEY] = pbc.unsqueeze(0)  # (1,3)
        ##############################

        # predict + extract data
        ### original code lines ###
        #out = self.model(AtomicData.to_AtomicDataDict(data))
        #forces = out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        #energy = out[AtomicDataDict.TOTAL_ENERGY_KEY].detach().cpu().item()
        ###########################
        #out = self.model(AtomicData.to_AtomicDataDict(data_dict))
        out = self.model(ad)

        # store results
        ### original code lines ###
        #self.results = {
        #    "energy": energy * self.energy_units_to_eV,
        #    # force has units eng / len:
        #    "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
        #}
        ### edit code lines ###
        self.results = {}
        # only store results the model actually computed to avoid KeyErrors
        if AtomicDataDict.TOTAL_ENERGY_KEY in out:
            self.results["energy"] = self.energy_units_to_eV * (
                out[AtomicDataDict.TOTAL_ENERGY_KEY]
                .detach()
                .cpu()
                .numpy()
                .reshape(tuple())
            )
            # "force consistant" energy
            self.results["free_energy"] = self.results["energy"]
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
            self.results["energies"] = self.energy_units_to_eV * (
                out[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                .detach()
                .squeeze(-1)
                .cpu()
                .numpy()
            )
        if AtomicDataDict.FORCE_KEY in out:
            # force has units eng / len:
            self.results["forces"] = (
                self.energy_units_to_eV / self.length_units_to_A
            ) * out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        if AtomicDataDict.STRESS_KEY in out:
            stress = out[AtomicDataDict.STRESS_KEY].detach().cpu().numpy()
            stress = stress.reshape(3, 3) * (
                self.energy_units_to_eV / self.length_units_to_A**3
            )
            # ase wants voigt format
            stress_voigt = full_3x3_to_voigt_6_stress(stress)
            self.results["stress"] = stress_voigt
        # Check for charges and add them if available
        charges_key = getattr(self, "charges_key", "charges")  # default value "charges"
        if charges_key in out:
            self.results["charges"] = out[charges_key].detach().cpu().numpy()
