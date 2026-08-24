# Dipole-informed Electrode Potential Network (DEPNet)

## DEPNet
DEPNet (Dipole-informed Electrode Potential Network) is a neural network model for predicting electrode potentials from atomic structures.  
The model uses atomic positions and chemical species as input and predicts the electrode potential through atom-wise representations.  

## Installation
1. Install [PyTorch](https://pytorch.org/get-started/locally/). In this branch, the package is tested on  
   - CUDA==13.0 (11.7, 11.3)
   - Python>=3.8 (3.12)
   - PyTorch==2.11.0 (1.13.0, 1.11.0)
   - Numpy >=2.0, <3.0 (2.4.3)
   - e3nn >= 0.6.0
```shell
python -m pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 -f https://download.pytorch.org/whl/torch_stable.html
```
2. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). **Note that this package does not work with the latest PyG's major version. Please install `torch_geometric<=1.7.2`.** For example,
```shell
Option 1 (CUDA 11.7)
export CUDA=cu117
export TORCH=1.13.0
python -m pip install torch-scatter==2.0.8 torch-sparse==0.6.11 torch-geometric==1.7.2 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html

Option 2 (CUDA 13.0)
pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -U torch-scatter torch-sparse torch-geometric==1.7.2 -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
```
3. Install other dependencies
```shell
python -m pip install -r requirements.txt
```
4. Install this package
```shell
python -m pip install .
```

## Usage

### Prepare datasets
Datasets should be included the potential drop.

### Train network
All settings for training are described with a YAML file. `depnet-train` command start to train a network.
```shell
depnet-train configs/dU_model_simple.yaml
```
Note that **`depnet-train` is assumed to be executed at the top of this reposity,** because a directory path for a dataset, `dataset_file_name` in the YAML file, may be relative.
The result are stored under `root` directory specified in the YAML file.

Example configurations are provided in [NequIP](https://github.com/mir-group/nequip#basic-network-training) [3-4], which this package is developed on the top of.
There are a few additional options for this package
```yaml
# in YAML file
model_builder: depnet.models.ElectrodePotentialModel  
pbc: true  # iff true, is periodic system  
use_electrodeU: true
```

### Deployed model & NequIP calculator
After the model train is complete, the deployed model can be generated from under command (it only works with our edited NequIP code).  
```shell
depnet-deploy build /your_working_dir/ /deployed_path/deploy_file_name 
```
When you generate a deployed model that works with GPU, type under command (additional option command).
```shell
depnet-deploy build /your_working_dir/ /deployed_path/deploy_file_name --device cuda
```
However, "pair" files for [Lammps](https://www.lammps.org/#gsc.tab=0) were unprepared yet.  
So, use the ASE calculator from NequIP, we provide the nequip_calculator example in the "util" directory.

### Resume from check-point file
Nequip 0.3.3 version doesn't support the "resume from check-point file".
Here, we prepared the updated file for training (depnet/scripts/train_resume-ckpt.py).
Before installing the package, use this updated code as follows:
```shell
cd depnet/scripts/
cp train_resume-ckpt.py train.py
```
Once you have overridden the original file (train.py) with the updated file (train_resume-ckpt.py), follow the installation guide.
```shell
cd ../../
pip install .
```
When you need to train MLP from check point after all installations, follow our guidance.
- Edit configure file (*.yaml)  
Add below tags.  
```
resume_from_ckpt: "../ckpt300.pth"   # check point file path
# resume_epoch: 300        # optional; if omitted, inferred from checkpoint filename
resume_strict: false
append: false              # recommended: use a new run folder (preserves previous logs)
```
- Command (do not use "restart"!)
```
depnet_train config.yaml
```


## Code base
DEPNet is developed based on the original [Charge_transfer_nnp](https://github.com/pfnet-research/charge_transfer_nnp/tree/master) implementation.  
The original code provides a NequIP-based framework with charge equilibration (Qeq). DEPNet modifies and extends this framework primarily for electrode-potential prediction.  


## References
1. Tsz Wai Ko, Jonas A. Finkler, Stefan Goedecker, Jörg Behler, A fourth-generation high-dimensional neural network potential with accurate electrostatics including non-local charge transfer, [Nat. Commun. 12, 398 (2021)](https://www.nature.com/articles/s41467-020-20427-2).
1. https://archive.materialscloud.org/record/2020.137
1. S. Batzner et al., [arxiv:2101.03164](https://arxiv.org/abs/2101.03164)
1. https://github.com/mir-group/nequip
