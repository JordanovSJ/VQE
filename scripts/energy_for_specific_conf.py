from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_elements import UCCGSD, UCCSD, DoubleExchange, SingleExchange, CustomDoubleExcitation
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

    # logging
    LogUtils.log_cofig()

    # uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)
    ansatz_elements = []
    ansatz_element_1 = CustomDoubleExcitation([4, 5], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    ansatz_elements.append(ansatz_element_1)
    ansatz_element_2 = CustomDoubleExcitation([2, 5], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    ansatz_elements.append(ansatz_element_2)
    ansatz_element_3 = CustomDoubleExcitation([2, 3], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    ansatz_elements.append(ansatz_element_3)
    ansatz_element_4 = CustomDoubleExcitation([3, 4], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    ansatz_elements.append(ansatz_element_4)
    ansatz_element_5 = CustomDoubleExcitation([6, 7], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    ansatz_elements.append(ansatz_element_5)
    ansatz_element_6 = CustomDoubleExcitation([8, 9], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    ansatz_elements.append(ansatz_element_6)
    ansatz_element_7 = SingleExchange(4, 10)
    ansatz_elements.append(ansatz_element_7)
    ansatz_element_8 = SingleExchange(5, 11)
    ansatz_elements.append(ansatz_element_8)
    ansatz_element_9 = SingleExchange(3, 11)
    ansatz_elements.append(ansatz_element_9)
    ansatz_element_10 = SingleExchange(2, 10)
    ansatz_elements.append(ansatz_element_10)
    ansatz_element_11 = CustomDoubleExcitation([0, 1], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    ansatz_elements.append(ansatz_element_11)

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r})

    # var_parameters = [0.24704637,  0.03216793, -0.03216734]
    #
    # energy = vqe_runner.get_energy(var_parameters, ansatz_elements)
    energy = vqe_runner.vqe_run()

    print(energy)
