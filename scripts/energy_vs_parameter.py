from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF, BeH2
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

    molecule = BeH2
    r =1.3264

    uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)
    ansatz_element_1 = DoubleExchangeAnsatzElement([4, 5], [10, 11])#, rescaled=True)
    # ansatz_element_1 = uccsd.get_double_excitation_list()[414]
    ansatz_elements = [ansatz_element_1]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, optimizer='Nelder-Mead')

    angles = numpy.arange(20)*numpy.pi/200
    energies = []
    for angle in angles:
        energies.append(vqe_runner.get_energy([angle], ansatz_elements))

    plt.plot(angles, energies)
    plt.xlabel(r'Angle, [$\pi$]')
    plt.ylabel('Energy, [Hartree]')
    plt.title('Mol.: {}, ansatz_element: {}'.format(molecule.name, ansatz_elements[0].element))
    plt.show()
