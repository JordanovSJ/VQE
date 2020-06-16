from src.vqe_runner import VQERunner
from src.q_systems import H2, LiH, HF, BeH2
from src.ansatz_element_lists import UCCGSD, UCCSD, DoubleExchange, SingleQubitExcitation, DoubleQubitExcitation
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

    uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)
    # ansatz_element_0 = DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, parity_dependence=False, d_exc_correction=False)

    ansatz_element_1 = DoubleQubitExcitation([4, 5], [10, 11])
    # ansatz_element_2 = DoubleExchange([3, 4], [10, 11], rescaled=True)
    # ansatz_element_3 = DoubleExchange([2, 3], [10, 11], rescaled=True)
    # ansatz_element_4 = DoubleExchange([6, 7], [10, 11], rescaled=True)
    # ansatz_element_5 = DoubleExchange([8, 9], [10, 11], rescaled=True)
    # ansatz_element_6 = DoubleExchange([1, 2], [10, 11], rescaled=True)
    # ansatz_element_7 = SingleExchange(5, 11)#, rescaled=True)

    # ansatz_element_1 = uccsd.get_double_excitation_list()[414]
    ansatz_elements = [ansatz_element_1] #ansatz_element_1, ansatz_element_2, ansatz_element_3, ansatz_element_4, ansatz_element_5,ansatz_element_6]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, optimizer='Nelder-Mead')

    angles = (numpy.arange(40) - 20)*numpy.pi/200 - 0.1
    energies = []
    for angle in angles:
        energies.append(vqe_runner.get_energy([angle],ansatz_elements))

    plt.plot(angles, energies)
    plt.xlabel(r'Angle, [$\pi$]')
    plt.ylabel('Energy, [Hartree]')
    plt.title('Mol.: {}, ansatz_element: {}: {}'.format(molecule.name, len(ansatz_elements), ansatz_elements[-1].element))
    plt.show()
