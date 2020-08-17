# Lantern: safer than a torch

Lantern is a software package to support SMT analysis of affine
multiplexing neural networks using PyTorch and Z3.

See: [docs/lantern/](docs/lantern/index.html) for (generated) documentation.

See: [requirements.txt](requirements.txt) for dependencies.

An annotated `example.py` demonstrates basic usage.

Lantern was developed against `torch==1.5.0` and `z3-solver==4.8.7.0`.
Linear Real Arithmetic appears to be stable, but we have noticed some changes
to the floating point solvers as z3 continues to be developed. To reproduce our
environment, we recommend executing:

    git clone https://github.com/JHUAPL/lantern-smt.git
    pip3 install -r requirements.txt
    python3 example.py

In a Python 3 virtual environment.
