# Radio Galaxy Classification Using a Neural Net Approach

This project aims to use a neural net to classify galaxies in the
[Radio Galaxy Dataset [1]](https://zenodo.org/records/7120632) created by Griese et al.
The goal is to find one of four classes, *FR-I*, *FR-II*, *compact*, or *bent*,
for each galaxy image.


## Dataset

The dataset is a collection of several catalogues using the FIRST radio galaxy
survey [[2]](https://ui.adsabs.harvard.edu/abs/1995ApJ...450..559B/abstract).
To these images, the following license applies:

"Provenance: The FIRST project team: R.J. Becker, D.H. Helfand, R.L. White M.D. Gregg. S.A. Laurent-Muehleisen.
Copyright: 1994, University of California. Permission is granted for publication and reproduction of this material
for scholarly, educational, and private non-commercial use. Inquiries for potential commercial uses should be
addressed to: Robert Becker, Physics Dept, University of California, Davis, CA  95616"


## Installation

To use the code in this repository, install the conda virtual environment found in
`environment.yml`. An installation of [Miniforge3](https://github.com/conda-forge/miniforge) is recommended due
to it providing a minimal installation of Python together with the fast and reliable mamba package
manager.

The environment can be installed via
```
$ mamba env create -f environment.yml
```

Additionally, you will need an installation of `cuda >= 12.1` to fully utilize pytorch.
Versions lower than `12.1` may work too, but have not been tested. Change the version
in the environment file depending on the version installed on your system.


## References
[1] Griese, F., Kummer, J., & Rustige, L., floriangriese/RadioGalaxyDataset: v0.1.1 (v0.1.1), Zenodo (2022). [https://doi.org/10.5281/zenodo.7120632](https://doi.org/10.5281/zenodo.7120632)

[2] R. H. Becker, R. L. White, D. J. Helfand, The FIRST Survey: Faint Images of the Radio Sky at Twenty Centimeters,
The Astrophysical Journal 450 (1995) 559.
