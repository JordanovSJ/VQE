from src.vqe_runner import VQERunner
from src.molecules.molecules import H2, LiH, HF, BeH2
from src.ansatz_element_sets import *
from src.backends import QiskitSimBackend
from src.utils import LogUtils
from src.iter_vqe_utils import *
from src.cache import *

import matplotlib.pyplot as plt

from openfermion import QubitOperator
import logging
import time
import numpy
import pandas
import datetime
import scipy
import qiskit
from functools import partial
import ast


if __name__ == "__main__":

    r = 1.316
    molecule = BeH2(r=r)  #frozen_els={'occupied': [0, 1], 'unoccupied': []})

    df = pandas.read_csv('../results/iter_vqe_results/vip/BeH2_g_adapt_gsdfe_comp_exc_06-Nov-2020.csv')

    state = DataUtils.ansatz_from_data_frame(df, molecule)
    ansatz = state.ansatz_elements

    u1_counts = []
    u1_count = 0
    cnot_counts = []
    cnot_count = 0
    for i, element in enumerate(ansatz):
        element_qubits = element.qubits
        if len(element_qubits[0]) == 1:
            new_element = SpinCompSFExc(*element_qubits[0], *element_qubits[1], molecule.n_orbitals, encoding='bk')
            element_qasm = new_element.get_qasm([0.1])
        else:
            new_element = SpinCompDFExc(element_qubits[0], element_qubits[1], molecule.n_orbitals, encoding='bk')
            element_qasm = new_element.get_qasm([0.1])

        gate_count = QasmUtils.gate_count_from_qasm(element_qasm, molecule.n_orbitals)
        cnot_count += gate_count['cnot_count']
        u1_count += gate_count['u1_count']
        cnot_counts.append(cnot_count)
        u1_counts.append(u1_count)

    df['bk_cnot_count'] = cnot_counts
    df['bk_u1_count'] = u1_counts

    df.to_csv('../results/iter_vqe_results/BeH2_adapt_fe_bk_n=10_r=1316_31-Mar-2021.csv')