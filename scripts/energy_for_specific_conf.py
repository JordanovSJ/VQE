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
        DoubleExchange([2, 5], [10, 13], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
        DoubleExchange([3, 4], [11, 12], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
        DoubleExchange([4, 5], [12, 13], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
        DoubleExchange([2, 3], [8, 9], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
        DoubleExchange([2, 3], [6, 7], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
        DoubleExchange([2, 5], [11, 12], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([3, 4], [10, 13], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([2, 3], [12, 13], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([3, 4], [11, 12], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=True,
                       parity_dependence=True),
        SingleExchange(5, 10), SingleExchange(2, 10), SingleExchange(3, 11)
        ]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r})

    var_parameters = var_parameters = [-0.006286855482474437, 0.02322160352844799, 0.01761525382660268, 0.01295646822009815,
                      0.01666999491692214, 0.02171421596304153, 0.021744136592131823, -0.017092625626106632,
                      -0.017067333762441125, 0.03798730875674191, 0.010456438059108284, 0.009035857484448868,
                      -0.0060881581645557915, -0.019570178119205736, -0.011190260456751982, 0.010990888659595147, 0.0]
    energy = vqe_runner.get_energy(var_parameters, ansatz_elements)

    print(energy)
