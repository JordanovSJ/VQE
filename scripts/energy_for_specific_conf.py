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
        DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=False, parity_dependence=True),
        DoubleExchange([2, 3], [10, 11], rescaled_parameter=True, d_exc_correction=False, parity_dependence=True),
        DoubleExchange([3, 4], [11, 12], rescaled_parameter=True, d_exc_correction=False, parity_dependence=True),
        DoubleExchange([4, 5], [12, 13], rescaled_parameter=True, d_exc_correction=False, parity_dependence=True),
        DoubleExchange([2, 3], [6, 7], rescaled_parameter=True, d_exc_correction=False, parity_dependence=True),
        DoubleExchange([3, 4], [10, 13], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        DoubleExchange([2, 5], [11, 12], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        DoubleExchange([2, 5], [10, 13], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        DoubleExchange([2, 3], [8, 9], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        DoubleExchange([2, 3], [12, 13], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        DoubleExchange([3, 5], [11, 13], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        DoubleExchange([2, 4], [10, 12], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        DoubleExchange([3, 4], [11, 12], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        SingleExchange(3, 12), SingleExchange(4, 11),
        DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        SingleExchange(2, 13),
        DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        DoubleExchange([3, 4], [10, 13], rescaled_parameter=True, d_exc_correction=False,
                       parity_dependence=True),
        SingleExchange(3, 13),
        ]

    # ansatz_elements = [
    #     DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
    #     DoubleExchange([2, 3], [10, 11], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
    #     DoubleExchange([2, 5], [10, 13], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
    #     DoubleExchange([3, 4], [11, 12], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
    #     DoubleExchange([4, 5], [12, 13], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
    #     DoubleExchange([2, 3], [8, 9], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([2, 3], [6, 7], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([2, 5], [11, 12], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([3, 4], [10, 13], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([2, 3], [12, 13], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([3, 4], [11, 12], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     SingleExchange(5, 10),
    #     SingleExchange(2, 10),
    #     SingleExchange(3, 11),
    #     DoubleExchange([0, 3], [8, 9], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([0, 3], [6, 7], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([1, 2], [8, 9], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([1, 2], [6, 7], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([0, 1], [6, 7], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     DoubleExchange([0, 1], [8, 9], rescaled_parameter=True, d_exc_correction=True,
    #                    parity_dependence=True),
    #     SingleExchange(3, 12)
    # ]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation,
                           molecule_geometry_params={'distance': r})

    # var_parameters = [-0.0065128523203437354, 0.02321899604703268, 0.01767897410373476, 0.013030636489892819,
    #                   0.016591824072023514, 0.021628392156845887, 0.021650230787297917, -0.017140504232529656,
    #                   -0.01710588486162598, 0.038206336858573886, 0.010456460058682412, 0.009490079862521126,
    #                   -0.00617656567425933, -0.019701873935962483, -0.011193926432043514, 0.01094580764824237,
    #                   -0.0014180166517939082, -0.0014196312906968578, 0.0012033041124553299, 0.0012017031176923489,
    #                   0.0006889241133224969, 0.0006843416243886342, -0.004472240537827521]

    var_parameters = [0.028669163026094503, 0.022537232912906303, 0.017489866500661134, 0.01629385690836846,
                      0.020851881457013468, -0.017647056790242338, -0.015327407983115066, 0.01789008153595026,
                      0.020923010546973268, 0.009497179062575096, 0.0024758813833710277, 0.0031049292770414087,
                      -0.0006, -0.0016353927583732481, -0.0072252648121858744,
                      -0.03436254148045186, -0.0011640242426753896, -0.0011606935508930922, -0.0085,
                      0.0024093251382888855, 0.00991045371533506]



    energy = vqe_runner.get_energy(var_parameters, ansatz_elements)

    print(energy)
