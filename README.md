# VQE
### This repository contains a code used to perform classical numerical simulations of molecules with the variational quantum eigensolver (VQE).The purpose of this repository is to be used as a benchmark to test and design ADAPT-VQE protocols (see Ref. 1).


* An implementation of the VQE is contained in src/vqe_runner.py
* The vqe_runner uses one of two different backends (in src/backends.py) to evaluate the expectation value of a quantum operator w.r.t. to a qubit state defined by an ansatz
* The ansatz is defined by a list of ansatz elements. Different types of ansatz elements are defined in src/ansatz_elements.py
* src/molecules/molecules.py contains a list of example molecular systems.

### Adapt-VQE protocol (see Refs. 1,2,3,4): 

* An implementation of the QEB-ADAPT-VQE protocol (Ref. 1) is contained in scripts/adapt_vqes/iqeb_vqe.py
* An implementation of the excited-QEB-ADAPT-VQE protocol (Ref. 3) is contained in scripts/adapt_vqes/excited_iqeb_vqe.py
* An implementation of the fermionic-ADAPT-VQE protocol (Ref. 2) is contained in scripts/adapt_vqes/adapt_vqe.py
* An implementation of the qubit-ADAPT-VQE protocol (Ref. 4) is contained in scripts/adapt_vqes/qubit_adapt_vqe.py

### References:

1. https://www.nature.com/articles/s42005-021-00730-0
2. https://www.nature.com/articles/s41467-019-10988-2
3. https://arxiv.org/abs/2106.06296
4. https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.020310

## Installation (Linux):

i. Install using the environment.yml file.
 - note that this option may not work for some dependencies

ii. If using the environment.yml file does not work, install manually:

1. Create a new conda environment


3. Install the following libraries:
  - qiskit 0.23.5 (or newer version)
  - numpy 1.20.0 (or newer version)
  - scipy 1.6.0
  - ray
  - pandas

    (Install the next three libraries in the given order. I recommend isntalling psi4
and openfermionpsi4 as given at https://github.com/quantumlib/OpenFermion-Psi4 )
  - psi4
  - openfermionpsi4 0.5 (or newer version)
  - openfermion 1.0.0 (or newer version)

c) Make sure that the following directories are added to PYTHONPATH:
 - "/home/ .. path/to .. /VQE"
 - "/home/ .. path/to .. /VQE/src"

(To add these paths permanently you can use the following command:
"conda develop '/home/ .. path/ to  .. /VQE/'")
