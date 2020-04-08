from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF, BeH2
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

    molecule = BeH2
    r =  1.316

    # logging
    LogUtils.log_cofig()

    # # uccsd = UCCSD(molecule.n_orbitals, molecule.n_electrons)
    # ansatz_elements = []
    # ansatz_element_1 = CustomDoubleExcitation([4, 5], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    # ansatz_elements.append(ansatz_element_1)
    # ansatz_element_2 = CustomDoubleExcitation([2, 5], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    # ansatz_elements.append(ansatz_element_2)
    # ansatz_element_3 = CustomDoubleExcitation([2, 3], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    # ansatz_elements.append(ansatz_element_3)
    # ansatz_element_4 = CustomDoubleExcitation([3, 4], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    # ansatz_elements.append(ansatz_element_4)
    # ansatz_element_5 = CustomDoubleExcitation([6, 7], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    # ansatz_elements.append(ansatz_element_5)
    # ansatz_element_6 = CustomDoubleExcitation([8, 9], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    # ansatz_elements.append(ansatz_element_6)
    # ansatz_element_7 = SingleExchange(4, 10)
    # ansatz_elements.append(ansatz_element_7)
    # ansatz_element_8 = SingleExchange(5, 11)
    # ansatz_elements.append(ansatz_element_8)
    # ansatz_element_9 = SingleExchange(3, 11)
    # ansatz_elements.append(ansatz_element_9)
    # ansatz_element_10 = SingleExchange(2, 10)
    # ansatz_elements.append(ansatz_element_10)
    # ansatz_element_11 = CustomDoubleExcitation([0, 1], [10, 11])#, rescaled_parameter=True, parity_dependence=True)
    # ansatz_elements.append(ansatz_element_11)

    ansatz_elements = [
        DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
        DoubleExchange([2, 3], [10, 11], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
        DoubleExchange([3, 4], [11, 12], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
        DoubleExchange([4, 5], [12, 13], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
        DoubleExchange([2, 3], [6, 7], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
        DoubleExchange([3, 4], [10, 13], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),

        DoubleExchange([2, 5], [11, 12], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([2, 5], [10, 13], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([2, 3], [8, 9], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([2, 3], [12, 13], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([3, 5], [11, 13], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([2, 4], [10, 12], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([3, 4], [11, 12], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        SingleExchange(3, 12)
        ]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r})

    var_parameters = [0.02542504343149286, 0.022800826074214103, 0.019658085166753464, 0.016359315630631654,
                      0.020856005034729494, -0.015290856033162612, -0.015482320072768922, 0.017327150133654004,
                      0.020917710627941843, 0.0089730955422877, 0.0030479645826446636, 0.003004512035924215,
                      -0.0015, -0.003517727107124633, -0.014030400855921022, 0.0]
    energy = vqe_runner.get_energy(var_parameters, ansatz_elements)

    print(energy)
