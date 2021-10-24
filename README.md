# VQE
### This repository contains a code used to perform classical numerical simulations of molecules with the variational quantum eigensolver (VQE).The purpose of this repository is to be used as a benchmark to test and design iterative VQE protocols (see Ref. 1).


* An implementation of the VQE is contained in src/vqe_runner.py
* The vqe_runner uses one of two different backends (in src/backends.py) to evaluate the expectation value of a quantum operator w.r.t. to a qubit state defined by an ansatz
* The ansatz is defined by a list of ansatz elements. Different types of ansatz elements are defined in src/ansatz_elements.py
* src/molecules/molecules.py contains a list of example molecular systems.

### Adapt-VQE protocol (see Refs. 1,2,3,4): 

* An implementation of the QEB-ADAPT-VQE protocol (Ref. 1) is contained in scripts/iter_vqe/iqeb_vqe.py
* An implementation of the excited-QEB-ADAPT-VQE protocol (Ref. 3) is contained in scripts/iter_vqe/excited_iqeb_vqe.py
* An implementation of the fermionic-ADAPT-VQE protocol (Ref. 2) is contained in scripts/iter_vqe/adapt_vqe.py
* An implementation of the qubit-ADAPT-VQE protocol (Ref. 4) is contained in scripts/iter_vqe/qubit_adapt_vqe.py

### References:

1. https://arxiv.org/abs/2011.10540
2. https://doi.org/10.1038/s41467-019-10988-2
3. https://arxiv.org/abs/2106.06296
4. https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.020310

## Installation:

1. Install using the environment.yml file.
 - note that this option might not work for some dependencies

2. Manual installation:
 i. Create a new conda environment
 ii. Install the following libraries:
  - qiskit 0.23.5 (or newer version)
  - openfermion 1.0.0 (or newer version)
  - openfermionpsi4 0.5
  - numpy 1.20.0 (or newer version)
  - scipy 1.6.0
  - ray
  - pandas 