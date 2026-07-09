from setuptools import find_packages, setup

setup(
    name="depnet",
    version="0.0.1",
    author="Jinwoong Chae",
    license="MIT",
    description="PyTorch implementation for charge transfer and electronstatic interactions",
    packages=find_packages(),
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "depnet-train = depnet.scripts.train:main",
            "depnet-requeue = depnet.scripts.requeue:main",
            "depnet-restart = depnet.scripts.restart:main",
            "depnet-deploy = depnet.scripts.deploy:main",
        ]
    },
    install_requires=[
        "torch>=1.8.0",
        "torch-geometric>=1.7.1",
        "nequip>=0.3.3",
        "numpy>=2.0,<3",
        "ase",
        "tqdm",
        "torch>=1.8",
        "torch_geometric>=1.7.1",
        "e3nn>=0.3.3",
        "pyyaml",
        "torch-runstats",
        "scikit-learn>=1.3",
    ],
)
