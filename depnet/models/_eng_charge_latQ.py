import logging

from e3nn.o3 import Irreps

from depnet.datasets import CHARGES_KEY, ELECTROSTATIC_ENERGY_KEY, TOTAL_CHARGE_KEY, STRESS_KEY, ELECTRODE_KEY, POISSON_KEY
from depnet.nn import (
    AttentionBlock,
    ChargeSkipConnection,
    ElectrostaticCorrection,
    Ewald,
    EwaldQeq,
    EwaldLatQ,
    Qeq,
    SumEnergies,
    TotalChargeEmbedding,
)
from depnet.data import AtomicDataDict
from depnet.nequip_nn import (
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
    ForceOutput,
    GraphModuleMixin,
    PerSpeciesScaleShift,
    SequentialGraphNetwork,
    StressForceOutput,
)
from depnet.nequip_nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)
import torch.nn as nn

# ===== Nonlinear NN
class NonLinearBlock(nn.Module, GraphModuleMixin):
    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=None,
        act_fn="Tanh",
        **kwargs,
    ):
        super().__init__()

        self.field = field
        self.out_field = out_field or field

        self._irreps_in = irreps_in
        in_irreps_all = Irreps(irreps_in[self.field])
        in_irreps_scalar = Irreps([ir for ir in in_irreps_all if ir.ir.l == 0])

        out_irreps = in_irreps_scalar
        self._irreps_out = dict(irreps_in)
        self._irreps_out[self.out_field] = out_irreps

        self.lin1 = AtomwiseLinear(
            irreps_in={self.field: in_irreps_scalar},
            irreps_out=out_irreps,
            field=self.field,
            out_field=self.field,
        )

        act_fn_dict = {
            "SiLU": nn.SiLU(),
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(),
            "Tanh": nn.Tanh(),
        }
        if act_fn not in act_fn_dict:
            raise ValueError(f"Unsupported activation: {act_fn}")
        self.act = act_fn_dict[act_fn]

        self.norm = nn.LayerNorm(out_irreps.dim)

        self.lin2 = AtomwiseLinear(
            irreps_in={self.field: out_irreps},
            irreps_out=out_irreps,
            field=self.field,
            out_field=self.field,
        )

    @property
    def irreps_in(self):
        return self._irreps_in

    @property
    def irreps_out(self):
        return self._irreps_out

    def forward(self, data):
        x = data[self.field]
        in_dim = self.lin1.linear.irreps_in.dim
        x = x[:, :in_dim]

        x = self.lin1({self.field: x})[self.field]
        x = self.act(x)
        x = self.norm(x)
        x = self.lin2({self.field: x})[self.field]

        data[self.out_field] = x
        return data
# =====

def EnergyChargeModel(**shared_params) -> SequentialGraphNetwork:
    """
    Energy-and-charge model architecture based on nequip.models._eng.EnergyModel
    """
    logging.debug("Start building the network model")

    #print("shared_params:", shared_params)

    num_layers = shared_params.pop("num_layers", 3)
    add_per_species_shift = shared_params.pop("PerSpeciesScaleShift_enable", False)
    pbc = shared_params.pop("pbc", False)

    use_charge = shared_params.pop("use_charge", False)
    use_qtot = shared_params.pop("use_qtot", False)  # enable if only use Q_tot
    use_ele = shared_params.pop("use_ele", False)
    use_qeq = shared_params.pop("use_qeq", False)
    if use_ele and (not use_charge):
        logging.info("Use ground-truth charges. Be careful what you did!")
    if use_qeq and (not use_charge):
        raise ValueError("Set use_charge: true to enable use_qeq: true !")
    if use_ele and use_qeq:
        raise ValueError("Set use_ele xor use_qeq for electrostatic correction!")

    # ===== latent Q option
    # use_latQ: use an MLP to predict latent charges q_i from node_features
    # and feed them to EwaldLatQ. This should be used without point-charge Ewald or Qeq.
    use_latQ = shared_params.pop("use_latQ", False)
    num_latQ_blocks = int(shared_params.pop("num_latQ_blocks", 1))
    if use_latQ:
        if (not use_charge) or use_ele or use_qeq:
            raise ValueError(
                "use_latQ: true requires use_charge=True, use_ele=False, use_qeq=False."
            )
        if num_latQ_blocks < 1:
            raise ValueError("num_latQ_blocks must be >= 1 when use_latQ: true.")
    # =====

    # ===== Solve Poisson from QEq + Ewald and predict electrode potential
    use_poisson = shared_params.pop("use_poisson", False)
    U_ref = shared_params.pop("U_ref", 0.0)
    #if use_poisson and (not use_qeq or use_latQ):
    #    raise ValueError("use_poisson: true requires use_qeq: true (QEq charges).")
    vac_top_frac = shared_params.pop("vac_top_frac", 0.15)
    dz_poisson = shared_params.pop("dz_poisson", 0.1)
    sigma_z = shared_params.pop("sigma_z", 0.5)
    elec_pad_A = shared_params.pop("elec_pad_A", 1.0)
    poisson_neutralize = shared_params.pop("poisson_neutralize", True)
    metal_species = shared_params.pop("metal_species", None)
    # =====

    use_nonlocal = shared_params.pop("use_nonlocal", False)
    energy_scale = shared_params.pop("_global_scale", 1.0)

    ### additional option tags ###
    non_linear_irreps_out = shared_params.get("conv_to_output_hidden_irreps_out", "1x0e")
    use_sc = shared_params.pop("use_sc", False) 
    linear_sc = shared_params.pop("linear_sc", False)
    use_dipole = shared_params.pop("use_dipole", False)
    use_bg = shared_params.pop("use_bg", False)
    use_slab = shared_params.pop("use_slab", False)
    use_electrodeU = shared_params.pop("use_electrodeU", False)
    ##############################

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    ### delecte total charge embedding line ###
    #if use_qtot or use_ele or use_qeq:
    #    # place TotalChargeEmbedding after chemical_embedding layer
    #    layers["total_charge_embedding"] = TotalChargeEmbedding
    ### end of editing line ###

    # TODO: nonlocal-interaction block in ConvNetLayer
    # add convnet layers
    # insertion preserves order
    for layer_i in range(num_layers):
        ### original code line ###
        #layers[f"layer{layer_i}_convnet"] = ConvNetLayer
        ### edited code line ###
        layers[f"layer{layer_i}_convnet"] = (
            ConvNetLayer,
            dict(use_sc=use_sc, linear_sc=linear_sc, debug=True),
        )
        ########################
        if use_nonlocal:
            layers[f"layer{layer_i}_attention"] = AttentionBlock

    layers["conv_to_output_hidden"] = AtomwiseLinear  ### original code line; move to case blocks

    # charge term
    if use_charge and use_ele:
        layers["atomic_charges"] = (
            AtomwiseLinear,
            dict(
                irreps_out="1x0e",
                field=AtomicDataDict.NODE_FEATURES_KEY,  # "node_features"
                out_field=CHARGES_KEY,
            ),
        )

    # Latent-Q branch: predict charges via a small MLP on top of node_features
    if use_charge and use_latQ:
        # NODE_FEATURES -> NonLinearBlock(MLP) -> latent_q_hidden
        for bi in range(num_latQ_blocks):
            layers_name = "latent_q_mlp" if bi == 0 else f"latent_q_mlp{bi}"
            layers[layers_name] = (
                NonLinearBlock,
                dict(
                    feature_irreps_hidden=non_linear_irreps_out,
                    field=AtomicDataDict.NODE_FEATURES_KEY if bi == 0 else "latent_q_hidden",
                    out_field="latent_q_hidden",
                    #out_field=AtomicDataDict.NODE_FEATURES_KEY,
                    act_fn="SiLU",
                ),
            )
        # latent_q_hidden -> Linear -> CHARGES_KEY (scalar q_i)
        layers["atomic_charges"] = (
            AtomwiseLinear,
            dict(
                irreps_out="1x0e",
                field="latent_q_hidden",
                #field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=CHARGES_KEY,
            ),
        )

    if use_ele:
        if pbc:
            # for periodic system, calculate electrostatic energy via Ewald summation
            layers["total_energy_with_ele"] = (
                Ewald,
                dict(scale=energy_scale),
            )
        else:
            # for nonperiodic system, calculate electrostatic energy directly
            layers["total_energy_with_ele"] = (
                ElectrostaticCorrection,
                dict(pbc=pbc, energy_scale=energy_scale),
            )
    elif use_qeq:
        # atomic charges are also generated in Qeq block
        if pbc:
            layers["total_energy_with_qeq"] = (
                EwaldQeq,
                dict(
                    scale=energy_scale,
                    use_dipole=use_dipole,
                    use_bg=use_bg,
                    use_slab=use_slab,
                    use_electrodeU=use_electrodeU,
                    use_poisson=use_poisson,
                    U_ref=U_ref,
                    vac_top_frac=vac_top_frac,
                    dz_poisson = dz_poisson,
                    sigma_z = sigma_z,
                    elec_pad_A = elec_pad_A,
                    poisson_neutralize = poisson_neutralize,
                    metal_species = metal_species,
                ),
            )
        else:
            layers["total_energy_with_qeq"] = (
                Qeq,
                dict(pbc=pbc, energy_scale=energy_scale),
            )
        
        # add charges to output-hidden
        layers["add_charges_to_output_hidden"] = (
            ChargeSkipConnection,
            dict(field="charges", debug=True) if pbc else dict(debug=False),
        )

    elif use_latQ:
        if not pbc:
            raise ValueError("use_latQ is currently only supported for pbc=True (Ewald).")
        # Bingqing-style latent Q branch:
        # use pre-predicted CHARGES_KEY as latent charges and compute long-range
        # electrostatic energy via EwaldLatQ (2D/3D Ewald, controlled by use_slab).
        layers["total_energy_with_latQ"] = (
            EwaldLatQ,
            dict(
                scale=energy_scale,
                use_dipole=use_dipole,
                use_bg=use_bg,
                use_slab=use_slab,
                use_electrodeU=use_electrodeU,
                use_poisson=use_poisson,
                U_ref=U_ref,
                vac_top_frac=vac_top_frac,
                dz_poisson = dz_poisson,
                sigma_z = sigma_z,
                elec_pad_A = elec_pad_A,
                poisson_neutralize = poisson_neutralize,
                metal_species = metal_species,
            ),
        )
        # NOTE: in latent-Q mode we do NOT feed charges back into node_features.
        # Short-range node_features remain the original ConvNet output.

    ### Nonlinear convNetLayer - additional code ###
    if (use_charge or use_ele or use_qeq or use_latQ): #and (not use_latQ):
        layers["hidden_non_linear"] = (
            ConvNetLayer,
            dict(feature_irreps_hidden=non_linear_irreps_out, use_sc=use_sc, linear_sc=linear_sc, debug=True),
        )
        '''
        layers["hidden_non_linear"] = (
            NonLinearBlock,
            dict(
                feature_irreps_hidden=non_linear_irreps_out,
                act_fn="Tanh",
            ),
        )
        '''
    ### end of additional code ###


    # short-range atomic energy
    layers["output_hidden_to_scalar"] = (
        AtomwiseLinear,
        dict(irreps_out="1x0e", out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY),
    )

    if add_per_species_shift:
        layers["per_species_scale_shift"] = (
            PerSpeciesScaleShift,
            dict(
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            ),
        )

    layers["total_energy_sum"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    )
    
    # add E_ele term into AtomicDataDict.TOTAL_ENERGY_KEY
    if use_ele or use_qeq or use_latQ:
        layers["sum_energy_terms"] = (
            SumEnergies,
            dict(
                input_fields=[AtomicDataDict.TOTAL_ENERGY_KEY, ELECTROSTATIC_ENERGY_KEY],
            ),
        )

    # additional irreps_in for charges
    '''
    irreps_in = {
        TOTAL_CHARGE_KEY: Irreps("1x0e"),  # total charge is scalar
    }
    if (not use_charge) and use_ele:
        # debug option
        irreps_in[CHARGES_KEY] = Irreps("1x0e")
    '''

    '''
    return SequentialGraphNetwork.from_parameters(
        shared_params=shared_params,
        layers=layers,
        irreps_in=irreps_in,
    )
    '''
    irreps_in = {}
    if use_charge or use_ele or use_qeq:
        irreps_in[TOTAL_CHARGE_KEY] = Irreps("1x0e")
        if (not use_charge) and use_ele:
            irreps_in[CHARGES_KEY] = Irreps("1x0e")
    # ===== "use_electrodeU" transfer to trainer.py
    model = SequentialGraphNetwork.from_parameters(
        shared_params=shared_params,
        layers=layers,
        irreps_in=irreps_in,
    )
    model.use_electrodeU = use_electrodeU
    return model
    # =====

def ForceChargeModel(**shared_params) -> GraphModuleMixin:
    """
    energy-charge-force model architecture based on nequip.models._eng.ForceModel
    """
    energy_charge_model = EnergyChargeModel(**shared_params)
    #return ForceOutput(energy_model=energy_charge_model)
    model = ForceOutput(energy_model=energy_charge_model)
    model.use_electrodeU = getattr(energy_charge_model, "use_electrodeU", False)
    return model

def StressChargeModel(**shared_params) -> GraphModuleMixin:
    energy_charge_model = EnergyChargeModel(**shared_params)
    #return StressForceOutput(model=energy_charge_model)
    model = ForceOutput(energy_model=energy_charge_model)
    model.use_electrodeU = getattr(energy_charge_model, "use_electrodeU", False)
    return model

def ElectrodePotentialModel(**shared_params) -> SequentialGraphNetwork:
    """
    Energy-and-charge model architecture based on nequip.models._eng.EnergyModel
    """
    logging.debug("Start building the network model for electrode potential")

    num_layers = shared_params.pop("num_layers", 3)
    add_per_species_shift = shared_params.pop("PerSpeciesScaleShift_enable", False)
    pbc = shared_params.pop("pbc", False)

    use_nonlocal = shared_params.pop("use_nonlocal", False)
    energy_scale = shared_params.pop("_global_scale", 1.0)

    ### additional option tags ###
    non_linear_irreps_out = shared_params.get("conv_to_output_hidden_irreps_out", "1x0e")
    use_sc = shared_params.pop("use_sc", False)
    linear_sc = shared_params.pop("linear_sc", False)
    use_dipole = shared_params.pop("use_dipole", False)
    use_bg = shared_params.pop("use_bg", False)
    use_slab = shared_params.pop("use_slab", False)
    use_electrodeU = shared_params.pop("use_electrodeU", False)
    ##############################

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    for layer_i in range(num_layers):
        layers[f"layer{layer_i}_convnet"] = (
            ConvNetLayer,
            dict(use_sc=use_sc, linear_sc=linear_sc, debug=True),
        )
        if use_nonlocal:
            layers[f"layer{layer_i}_attention"] = AttentionBlock

    layers["conv_to_output_hidden"] = AtomwiseLinear
    
    layers["U_output"] = (
        AtomwiseLinear,
        dict(
            irreps_out="1x0e",
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field="per_atom_U",
        ),
    )

    if add_per_species_shift:
        layers["U_per_species_scale_shift"] = (
            PerSpeciesScaleShift,
            dict(
                field="per_atom_U",
                out_field="per_atom_U",
            ),
        )

    layers["U_reduce"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field="per_atom_U",
            out_field="U_sum",
        ),
    )

    irreps_in = {}
    # ===== "use_electrodeU" transfer to trainer.py
    model = SequentialGraphNetwork.from_parameters(
        shared_params=shared_params,
        layers=layers,
        irreps_in=irreps_in,
    )
    model.use_electrodeU = use_electrodeU
    return model
    # =====
