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

    molecule = H2
    r = 0.735
    max_n_iterations = 2000

    # logging
    LogUtils.log_cofig()

    uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)
    # ansatz_element_1 = DoubleExchangeAnsatzElement([4, 5], [10, 11])
    ansatz_element_1 = DoubleExchangeAnsatzElement([0, 1], [2, 3])
    ansatz_elements = [ansatz_element_1]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, optimizer='Nelder-Mead')

    angles = numpy.arange(51)*numpy.pi/50
    energies = []
    for angle in angles:
        energies.append(vqe_runner.get_energy([angle], ansatz_elements))

    plt.plot(angles, energies)
    plt.xlabel(r'Angle, [$\pi$]')
    plt.ylabel('Energy, [Hartree]')
    plt.show()
