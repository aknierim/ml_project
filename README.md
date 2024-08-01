# *R*adi*O* ga*L*axy classi*F*ication using a ResNet (ROLF)

This project aims to use a neural net to classify galaxies in the
[Radio Galaxy Dataset [1]](https://zenodo.org/records/7120632) created by Griese et al.
The goal is to find one of four classes, *FR-I*, *FR-II*, *compact*, or *bent*,
for each galaxy image.

## Table of Contents

* [Dataset](#dataset)
* [Installation and Usage](#installation-and-usage)
  * [Installation](#installation)
  * [Usage](#usage)
    * [Data Download and Unpacking](#data-download-and-unpacking)
    * [Training](#training)
    * [Hyperparameter Optimization](#hyperparameter-optimization)
  * [Pre-trained Models](#pre-trained-models)
  * [Monitoring](#monitoring)
* [References](#references)

## Dataset

The dataset is a collection of several catalogues using the FIRST radio galaxy
survey [[2]](https://ui.adsabs.harvard.edu/abs/1995ApJ...450..559B/abstract).
To these images, the following license applies:

"Provenance: The FIRST project team: R.J. Becker, D.H. Helfand, R.L. White M.D. Gregg. S.A. Laurent-Muehleisen.
Copyright: 1994, University of California. Permission is granted for publication and reproduction of this material
for scholarly, educational, and private non-commercial use. Inquiries for potential commercial uses should be
addressed to: Robert Becker, Physics Dept, University of California, Davis, CA 95616"


## Installation and Usage

### Installation

To use the code in this repository, install the conda virtual environment found in
`environment.yml`. An installation of [Miniforge3](https://github.com/conda-forge/miniforge) is recommended since
it provides a minimal installation of Python together with the fast and reliable mamba package
manager.

The environment can be installed via
```
$ mamba env create -f environment.yml
```
and can be activated by calling
```
$ mamba activate rolf
```

A dev version of the environment can be installed via the file `environment_dev.yml`
After you installed and activated the environment, please install the `rolf` package
using [pip](https://pypi.org/project/pip/)
```
$ pip install -e .
```

Additionally, you will need an installation of `cuda >= 12.1` to fully utilize PyTorch.
Versions lower than `12.1` may work too, but have not been tested. Change the version
in the environment file depending on the version installed on your system.

### Usage
Make sure you have installed the environment and `rolf` and have activated it
call
```
$ rolf-info
```
to print an overview of available commands.
```
$ rolf-info --tools
```
will print an overview of all available command-line interface (CLI) tools.

#### Data Download and Unpacking
Before any model can be trained, please call
```
$ rolf-data -n -o build
```
in the root directory of the repository. This will download the data from the
list of URLs in `urls.toml`. Then call
```
$ rolf-unpack build/galaxy_data_h5.zip -o data
```
and
```
$ rolf-unpack build/galaxy_data.zip -o data
```
to unpack the data to the data directory. The directories are created automatically.

#### Training
Config files for both the training and the hyperparameter optimization are provided in the `configs` directory
and are loaded per default when calling any of the following CLI tools.
- The training of ROLF can be started using
```
$ rolf-train
```
- The training of the random forest classifier ROMF can be started using
```
$ romf-train
```

#### Hyperparameter Optimization
- The hyperparameter optimization for ROLF can be started via
```
$ rolf-optim
```
- The hyperparameter optimization of the random forest classifier can be started by calling
```
$ romf-optim
```
All optional arguments of the CLI tools can be printed by adding the `--help` flag.

### Pre-trained Models
We have provided pre-trained models in the `trained_models` directory.
The ROLF model can be loaded and evaluated using the `classification_viewer.ipynb` notebook
found in the `notebooks` directory. This notebook can also be used to evaluate checkpoints
of models trained via `rolf-train`.

### Monitoring
The training progress of ROLF can be monitored using [TensorBoard](https://www.tensorflow.org/tensorboard):
```
$ tensorboard --logdir <path/to/checkpoints/directory>
```
The hyperparameter optimization can be monitored using [Optuna Dashboard](https://optuna-dashboard.readthedocs.io/en/latest/):
```
$ optuna-dashboard <path/to/database.sqlite3>
```

## References
[1] Griese, F., Kummer, J., & Rustige, L., ***"Radio Galaxy Dataset (v0.1.3)"***, Zenodo (2022).
[https://zenodo.org/doi/10.5281/zenodo.7113623](https://zenodo.org/doi/10.5281/zenodo.7113623)

[2] R. H. Becker, R. L. White, D. J. Helfand, ***"The FIRST Survey: Faint Images of the Radio Sky at Twenty Centimeters"***,
The Astrophysical Journal, Vol. 450, p. 559 (1995).
