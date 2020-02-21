from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_elements import UCCGSD, UCCSD, DoubleExchangeAnsatzElement
from src.backends import QiskitSimulation
from src.utils import LogUtils

import matplotlib.pyplot as plt

import logging
import time
import numpy
import pandas
import datetime
import scipy
import qiskit
from functools import partial


if __name__ == "__main__":

    molecule = HF
    r = 0.995
    max_n_iterations = 2000

    # logging
    LogUtils.log_cofig()

    uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)
    ansatz_element_1 = DoubleExchangeAnsatzElement([4, 5], [10, 11])
    ansatz_element_2 = DoubleExchangeAnsatzElement([2, 3], [10, 11])

    ansatz_elements = [ansatz_element_1, ansatz_element_2]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, optimizer='Nelder-Mead')

    var_parameters = [0.25964804, 0]

    energy = vqe_runner.get_energy(var_parameters, ansatz_elements)

    print(energy)
