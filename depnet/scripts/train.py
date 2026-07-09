""" Train a network."""
import argparse
import logging
from typing import Callable, Union

import e3nn
import e3nn.util.jit

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401
import torch
import yaml
from sklearn.linear_model import LinearRegression
from torch_scatter import scatter

from depnet.datasets import CHARGES_KEY
from depnet.datasets import ELECTRODE_KEY
#from nequip.data import AtomicDataDict
from depnet.data import AtomicDataDict
#from nequip.nn import RescaleOutput
from depnet.nequip_nn import RescaleOutput
#from nequip.utils import Config, dataset_from_config
from depnet.utils import Config, dataset_from_config
#from nequip.utils.test import assert_AtomicData_equivariant, set_irreps_debug
from depnet.nequip_utils.test import assert_AtomicData_equivariant, set_irreps_debug

default_config = dict(
    requeue=False,
    wandb=False,
    wandb_project="NequIP",
    wandb_resume=False,
    compile_model=False,
    model_builder="depnet.models.ForceModel",
    model_initializers=[],
    dataset_statistics_stride=1,
    default_dtype="float32",
    allow_tf32=True,
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
)


def main(args=None):
    fresh_start(parse_command_line(args))


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(description="Train a NequIP model.")
    parser.add_argument("config", help="configuration file")
    parser.add_argument(
        "--equivariance-test",
        help="test the model's equivariance before training",
        action="store_true",
    )
    parser.add_argument(
        "--model-debug-mode",
        help="enable model debug mode, which can sometimes give much more useful error messages at the cost of some speed. Do not use for production training!",
        action="store_true",
    )
    parser.add_argument(
        "--grad-anomaly-mode",
        help="enable PyTorch autograd anomaly mode to debug NaN gradients. Do not use for production training!",
        action="store_true",
    )
    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)
    for flag in ("model_debug_mode", "equivariance_test", "grad_anomaly_mode"):
        config[flag] = getattr(args, flag) or config[flag]

    return config


def _load_callable(obj: Union[str, Callable]) -> Callable:
    if callable(obj):
        pass
    elif isinstance(obj, str):
        obj = yaml.load(f"!!python/name:{obj}", Loader=yaml.Loader)
    else:
        raise TypeError
    assert callable(obj), f"{obj} isn't callable"
    return obj


def regression_by_species(dataset, train_idcs):
    atomic_numbers = torch.unique(dataset.data.atomic_numbers, sorted=True)
    counts = []
    for z in atomic_numbers:
        count_z = scatter(
            (dataset.data.atomic_numbers == z).to(torch.float32), dataset.data.batch, reduce="sum"
        )
        counts.append(count_z)
    X_all = torch.stack(counts).detach().numpy().T  # (num_samples, len(atomic_numbers))
    y_all = dataset.data.total_energy.detach().numpy()[:, 0]  # (num_samples, )

    X_train = X_all[train_idcs.numpy()]
    y_train = y_all[train_idcs.numpy()]
    reg = LinearRegression().fit(X_train, y_train)

    y_diff = y_all - reg.predict(X_all)
    device = dataset.data.pos.device
    dataset.data.total_energy = torch.unsqueeze(torch.tensor(y_diff, device=device), dim=1)

    return dataset


def fresh_start(config):
    # = Set global state =
    # Set TF32 support
    # See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available():
        if torch.torch.backends.cuda.matmul.allow_tf32 and not config.allow_tf32:
            # it is enabled, and we dont want it to, so disable:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    if config.model_debug_mode:
        set_irreps_debug(enabled=True)
    torch.set_default_dtype(
        {"float32": torch.float32, "float64": torch.float64}[config.default_dtype]
    )
    if config.grad_anomaly_mode:
        torch.autograd.set_detect_anomaly(True)

    e3nn.set_optimization_defaults(**config.get("e3nn_optimization_defaults", {}))

    # = Make the trainer =
    if config.wandb:
        import wandb  # noqa: F401

        from depnet.train.trainer_wandb import TrainerWandB

        # download parameters from wandb in case of sweeping
        from depnet.nequip_utils.wandb import init_n_update

        config = init_n_update(config)

        trainer = TrainerWandB(model=None, **dict(config))
    else:
        #from nequip.train.trainer import Trainer
        from depnet.train.trainer import Trainer

        trainer = Trainer(model=None, **dict(config))

    output = trainer.output
    config.update(output.updated_dict())

    # ===== separate train/val files
    use_sep    = bool(config.get("use_separate_files", False))
    train_path = config.get("dataset_train_name", None)
    valid_path = config.get("dataset_valid_name", None)

    if use_sep and train_path and valid_path:
        cfg_t = Config(dict(config))
        cfg_v = Config(dict(config))

        cfg_t["dataset_file_name"] = train_path
        cfg_v["dataset_file_name"] = valid_path

        if "extra_fixed_fields" not in cfg_t:
            cfg_t["extra_fixed_fields"] = {}
        if "extra_fixed_fields" not in cfg_v:
            cfg_v["extra_fixed_fields"] = {}
        cfg_t["extra_fixed_fields"]["cache_by_filename"] = True
        cfg_v["extra_fixed_fields"]["cache_by_filename"] = True

        train_ds = dataset_from_config(cfg_t)
        valid_ds = dataset_from_config(cfg_v)

        def _take_first_n(ds, n, do_shuffle):
            if n is None:
                return ds
            n = int(n); total = len(ds)
            if n > total:
                logging.warning(f"[warn] requested n={n} > len(dataset)={total}; using all {total}")
                n = total
            if do_shuffle:
                idx = torch.randperm(total)[:n].tolist()
            else:
                idx = torch.arange(total)[:n].tolist()
            return ds.index_select(idx)

        train_ds = _take_first_n(train_ds, getattr(trainer, "n_train", None), do_shuffle=trainer.shuffle)
        valid_ds = _take_first_n(valid_ds, getattr(trainer, "n_val",   None), do_shuffle=trainer.shuffle)

        trainer.set_dataset((train_ds, valid_ds))
        did_set_dataset = True
# =====

    # ===== original code block
    '''
    # = Load the dataset =
    dataset = dataset_from_config(config)
    logging.info(f"Successfully loaded the data set of type {dataset}...")

    # split train/test before calling trainer.set_dataset for energy regression
    total_n = len(dataset)
    if trainer.train_val_split == "random":
        idcs = torch.randperm(total_n)
    elif trainer.train_val_split == "sequential":
        idcs = torch.arange(total_n)
    else:
        raise NotImplementedError(f"splitting mode {trainer.train_val_split} not implemented")
    trainer.train_idcs = idcs[: trainer.n_train]
    trainer.val_idcs = idcs[trainer.n_train : trainer.n_train + trainer.n_val]
    # energy regression by species
    if config.get("regression_by_species", False):
        dataset = regression_by_species(dataset, trainer.train_idcs)

    # = Train/test split =
    trainer.set_dataset(dataset)
    '''
    # =====
    # ===== edited code block from original code block
    if not did_set_dataset:
        dataset = dataset_from_config(config)
        logging.info(f"Successfully loaded the data set of type {dataset}...")
        
        total_n = len(dataset)
        if trainer.train_val_split == "random":
            idcs = torch.randperm(total_n)
        elif trainer.train_val_split == "sequential":
            idcs = torch.arange(total_n)
        else:
            raise NotImplementedError(f"splitting mode {trainer.train_val_split} not implemented")
        trainer.train_idcs = idcs[: trainer.n_train]
        trainer.val_idcs = idcs[trainer.n_train : trainer.n_train + trainer.n_val]

        if config.get("regression_by_species", False):
            dataset = regression_by_species(dataset, trainer.train_idcs)

        trainer.set_dataset(dataset)
    # =====
    print(f"[dataset] train={len(trainer.dataset_train)}, val={len(trainer.dataset_val)}")

    # = Determine training type =
    train_on = config.loss_coeffs
    train_on = [train_on] if isinstance(train_on, str) else train_on
    train_on = set(train_on)
    force_training = "forces" in train_on
    charge_training = "charges" in train_on
    elec_U_training = "electrode_potential" in train_on   # ===== additional tag for electrode U training
    logging.debug(f"Force training mode: {force_training}")
    logging.debug(f"Charge training mode: {charge_training}")
    logging.debug(f"Electrode_U training mode: {elec_U_training}")
    del train_on

    # = Get statistics of training dataset =
    stats_fields = [
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.ATOMIC_NUMBERS_KEY,
    ]
    stats_modes = ["mean_std", "count"]
    if force_training:
        stats_fields.append(AtomicDataDict.FORCE_KEY)
        stats_modes.append("rms")
    if charge_training:
        stats_fields.append(CHARGES_KEY)
        stats_modes.append("rms")
    # ===== additional block: electrode U training
    if elec_U_training:
        stats_fields.append(ELECTRODE_KEY)
        stats_modes.append("rms")
    # =====

    stats = trainer.dataset_train.statistics(
        fields=stats_fields, modes=stats_modes, stride=config.dataset_statistics_stride
    )
    (
        (energies_mean, energies_std),
        (allowed_species, Z_count),
    ) = stats[:2]
    if force_training:
        # Scale by the force std instead
        force_rms = stats[2][0]
    # ===== additional block: electrode U training
    if elec_U_training:
        elec_U_rms = stats[-1][0]
    # =====
    del stats_modes
    del stats_fields

    config.update(dict(allowed_species=allowed_species))

    # = Determine shifts, scales =
    # This is a bit awkward, but necessary for there to be a value
    # in the config that signals "use dataset"
    global_shift = config.get("global_rescale_shift", "dataset_energy_mean")
    if global_shift == "dataset_energy_mean":
        global_shift = energies_mean
    elif (
        global_shift is None
        or isinstance(global_shift, float)
        or isinstance(global_shift, torch.Tensor)
    ):
        # valid values
        pass
    else:
        raise ValueError(f"Invalid global shift `{global_shift}`")

    global_scale = config.get(
        "global_rescale_scale", force_rms if force_training else energies_std
    )
    if global_scale == "dataset_energy_std":
        global_scale = energies_std
    elif global_scale == "dataset_force_rms":
        if not force_training:
            raise ValueError(
                "Cannot have global_scale = 'dataset_force_rms' without force training"
            )
        global_scale = force_rms
    elif (
        global_scale is None
        or isinstance(global_scale, float)
        or isinstance(global_scale, torch.Tensor)
    ):
        # valid values
        pass
    else:
        raise ValueError(f"Invalid global scale `{global_scale}`")

    RESCALE_THRESHOLD = 1e-6
    if isinstance(global_scale, float) and global_scale < RESCALE_THRESHOLD:
        raise ValueError(
            f"Global energy scaling was very low: {global_scale}. If dataset values were used, does the dataset contain insufficient variation? Maybe try disabling global scaling with global_scale=None."
        )
        # TODO: offer option to disable rescaling?

    logging.debug(
        f"Initially outputs are scaled by: {global_scale}, eneriges are shifted by {global_shift}."
    )

    # dirty hack to remember global_scale i.e. std of energy
    # Use it in electrostatic correction
    config["_global_scale"]: float = global_scale.item()

    # = Build a model =
    model_builder = _load_callable(config.model_builder)
    core_model = model_builder(**dict(config))

    # = Reinit if wanted =
    with torch.no_grad():
        for initer in config.model_initializers:
            initer = _load_callable(initer)
            core_model.apply(initer)

    # == Build the model ==
    # Currently we do not rescale atomic charges
    ''' original code block
    final_model = RescaleOutput(
        model=core_model,
        scale_keys=[AtomicDataDict.TOTAL_ENERGY_KEY]
        + ([AtomicDataDict.FORCE_KEY] if AtomicDataDict.FORCE_KEY in core_model.irreps_out else [])
        + ([AtomicDataDict.STRESS_KEY] if AtomicDataDict.STRESS_KEY in core_model.irreps_out else [])   ### additional code line
        + ([AtomicDataDict.PER_ATOM_ENERGY_KEY] if AtomicDataDict.PER_ATOM_ENERGY_KEY in core_model.irreps_out else [])   ### additional code line
        + ([CHARGES_KEY] if CHARGES_KEY in core_model.irreps_out else [])   ### additional code line
        + ([ELECTRODE_KEY] if ELECTRODE_KEY in core_model.irreps_out else []),   ### additional code line
        scale_by=global_scale,
        shift_keys=AtomicDataDict.TOTAL_ENERGY_KEY,
        shift_by=global_shift,
        trainable_global_rescale_shift=config.get("trainable_global_rescale_shift", False),
        trainable_global_rescale_scale=config.get("trainable_global_rescale_scale", False),
    )
    '''

    # ===== for electrode potential model
    shift_keys = []
    if AtomicDataDict.TOTAL_ENERGY_KEY in core_model.irreps_out:
        shift_keys.append(AtomicDataDict.TOTAL_ENERGY_KEY)

    scale_keys = []
    if AtomicDataDict.TOTAL_ENERGY_KEY in core_model.irreps_out:
        scale_keys.append(AtomicDataDict.TOTAL_ENERGY_KEY)
    if AtomicDataDict.FORCE_KEY in core_model.irreps_out:
        scale_keys.append(AtomicDataDict.FORCE_KEY)
    if AtomicDataDict.STRESS_KEY in core_model.irreps_out:
        scale_keys.append(AtomicDataDict.STRESS_KEY)
    if AtomicDataDict.PER_ATOM_ENERGY_KEY in core_model.irreps_out:
        scale_keys.append(AtomicDataDict.PER_ATOM_ENERGY_KEY)
    if CHARGES_KEY in core_model.irreps_out:
        scale_keys.append(CHARGES_KEY)
    if ELECTRODE_KEY in core_model.irreps_out:
        scale_keys.append(ELECTRODE_KEY)

    # === Build the model
    final_model = RescaleOutput(
        model=core_model,
        scale_keys=scale_keys,
        shift_keys=shift_keys,
        scale_by=global_scale,
        shift_by=global_shift,
        trainable_global_rescale_shift=config.get("trainable_global_rescale_shift", False),
        trainable_global_rescale_scale=config.get("trainable_global_rescale_scale", False),
    )
    # =====

    logging.info("Successfully built the network...")

    if config.compile_model:
        final_model = e3nn.util.jit.script(final_model)
        logging.info("Successfully compiled model...")

    # Record final config
    with open(output.generate_file("config_final.yaml"), "w+") as fp:
        yaml.dump(dict(config), fp)

    # Equivar test
    if config.equivariance_test:
        equivar_err = assert_AtomicData_equivariant(final_model, dataset.get(0))
        errstr = "\n".join(
            f"    parity_k={parity_k.item()}, did_translate={did_trans} -> max componentwise error={err.item()}"
            for (parity_k, did_trans), err in equivar_err.items()
        )
        del equivar_err
        logging.info(f"Equivariance test passed; equivariance errors:\n{errstr}")
        del errstr

    # Set the trainer
    trainer.model = final_model

    # ===== Resume from an arbitrary checkpoint
    import os, re, torch.nn as nn

    ckpt = config.get("resume_from_ckpt", None)
    if ckpt and os.path.isfile(ckpt):
        obj = torch.load(ckpt, map_location="cpu")
        
        # 1) model weight
        if isinstance(obj, dict) and "model" in obj:
            weights = obj["model"]
        elif isinstance(obj, dict) and "state_dict" in obj:
            weights = obj["state_dict"]
        elif isinstance(obj, nn.Module):
            weights = obj.state_dict()
        else:
            weights = obj

        missing, unexpected = trainer.model.load_state_dict(
            weights, strict=bool(config.get("resume_strict", False))
        )
        if missing or unexpected:
            logging.warning(f"[resume] missing={missing}, unexpected={unexpected}")
        
        # 2) optimizer/scheduler/scaler
        trainer.init()

        if isinstance(obj, dict) and "optimizer" in obj and getattr(trainer, "optim", None) is not None:
            trainer.optim.load_state_dict(obj["optimizer"])
            if "learning_rate" in config:
                for g in trainer.optim.param_groups:
                    g["lr"] = float(config["learning_rate"])

        # 3) resume
        m = re.search(r"(\d+)", os.path.basename(ckpt))
        start_ep = int(obj.get("epoch", config.get("resume_epoch", m.group(1) if m else 0))) \
                if isinstance(obj, dict) else int(config.get("resume_epoch", m.group(1) if m else 0))
        trainer.restart = True
        trainer.iepoch  = start_ep
        trainer.best_epoch = start_ep
        trainer.best_val_metrics = float("inf")
    # ===== end resume block

    # Train
    trainer.train()

    return

if __name__ == "__main__":
    main()
