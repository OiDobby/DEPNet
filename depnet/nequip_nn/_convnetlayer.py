from typing import Dict, Callable
import torch
import logging

from e3nn import o3
from e3nn.nn import Gate, NormActivation

from depnet.data import AtomicDataDict
from depnet.nequip_nn import (
    GraphModuleMixin,
    InteractionBlock,
)
from depnet.nequip_nn.nonlinearities import ShiftedSoftPlus
from depnet.nequip_utils.tp_utils import tp_path_exists


acts = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": ShiftedSoftPlus,
    "silu": torch.nn.functional.silu,
}


class ConvNetLayer(GraphModuleMixin, torch.nn.Module):
    """
    Args:

    """

    resnet: bool

    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        convolution=InteractionBlock,
        convolution_kwargs: dict = {},
        use_sc: bool = False, ### additaional tag
        linear_sc: bool = False, ### additaional tag
        num_layers: int = 3,
        resnet: bool = True,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
        debug=False,
    ):
        super().__init__()
        # initialization
        assert nonlinearity_type in ("gate", "norm")
        # make the nonlin dicts from parity ints instead of convinience strs
        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        self.feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        self.resnet = resnet
        self.num_layers = num_layers
        self.use_sc = use_sc ### additaional tag
        self.linear_sc = linear_sc ### additaional tag

        # We'll set irreps_out later when we know them
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.NODE_FEATURES_KEY],
        )

        edge_attr_irreps = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY]
        irreps_layer_out_prev = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]

        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l == 0
                and tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, ir)
            ]
        )

        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l > 0
                and tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, ir)
            ]
        )

        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        if nonlinearity_type == "gate":
            ir = (
                "0e"
                if tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, "0e")
                else "0o"
            )
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            # TO DO, it's not that safe to directly use the
            # dictionary
            equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    acts[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[acts[nonlinearity_gates[ir.p]] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

            conv_irreps_out = equivariant_nonlin.irreps_in.simplify()

        else:
            conv_irreps_out = irreps_layer_out.simplify()

            equivariant_nonlin = NormActivation(
                irreps_in=conv_irreps_out,
                # norm is an even scalar, so use nonlinearity_scalars[1]
                scalar_nonlinearity=acts[nonlinearity_scalars[1]],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )

        self.equivariant_nonlin = equivariant_nonlin

        # TODO: partial resnet?
        if irreps_layer_out == irreps_layer_out_prev and resnet:
            # We are doing resnet updates and can for this layer
            self.resnet = True
        else:
            self.resnet = False

        # TODO: last convolution should go to explicit irreps out
        logging.debug(
            f" parameters used to initialize {convolution.__name__}={convolution_kwargs}"
        )

        # override defaults for irreps:
        convolution_kwargs.pop("irreps_in", None)
        convolution_kwargs.pop("irreps_out", None)
        convolution_kwargs.pop("use_sc", None) ### additional line
        convolution_kwargs.pop("linear_sc", None) ### additional line
        self.conv = convolution(
            irreps_in=self.irreps_in,
            irreps_out=conv_irreps_out,
            use_sc=use_sc, ### additional option
            linear_sc=linear_sc, ### additional option
            **convolution_kwargs,
        )

        # The output features are whatever we got in
        # updated with whatever the convolution outputs (which is a full graph module)
        self.irreps_out.update(self.conv.irreps_out)
        # but with the features updated by the nonlinearity
        self.irreps_out[
            AtomicDataDict.NODE_FEATURES_KEY
        ] = self.equivariant_nonlin.irreps_out
        self.debug = debug

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # save old features for resnet
        old_x = data[AtomicDataDict.NODE_FEATURES_KEY]
        #print(f"Input irreps - Before Conv: {self.irreps_in}")
        #if self.debug:
            #print(f"Input shape - Before Conv: {data[AtomicDataDict.NODE_FEATURES_KEY].shape}")
            #print(f"Input irreps - Before Conv: {self.irreps_in}")
        # run convolution
        data = self.conv(data)
        # do nonlinearity
        data[AtomicDataDict.NODE_FEATURES_KEY] = self.equivariant_nonlin(
            data[AtomicDataDict.NODE_FEATURES_KEY]
        )
        # do resnet
        if self.resnet:
            data[AtomicDataDict.NODE_FEATURES_KEY] = (
                old_x + data[AtomicDataDict.NODE_FEATURES_KEY]
            )
        #print(f"Output irreps - After Conv: {self.irreps_out}")
        #if self.debug:
            #print("ConvNetLayer - After Conv:", data[AtomicDataDict.NODE_FEATURES_KEY].shape)
            #print(data[AtomicDataDict.NODE_FEATURES_KEY])
            #print(f"Output shape - After Conv: {data[AtomicDataDict.NODE_FEATURES_KEY].shape}")
            #print(f"Output irreps - After Conv: {self.irreps_out}")
        return data
