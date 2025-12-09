# Petrov-Galerkin addon for FEniCSx

The documentation can be found [here](https://mfeuerle.github.io/pgfenicsx/).

For installation run

```bash
$ conda env create -f environment.yml
$ conda activate pgfenicsx
$ pip install .
```

for an editable installation

```bash
$ pip install -e .
```

or simply copy the contents from `pgfenicsx/_pgfenicsx.py` to your local project.

To build the docs also run

```bash
$ conda env update -f docs/environment-sphinx.yml
$ cd docs
$ make html
```
