import logging
import time
import numpy
import pandas
import ray
import pandas
import datetime

import sys
import ast
sys.path.append('../../')

from src.vqe_runner import VQERunner
from src.q_system import *
from src.ansatz_element_sets import *
from src.backends import QiskitSimBackend
from src.utils import LogUtils
from src.iter_vqe_utils import *
from src.cache import *
from src.molecules.molecules import *


if __name__ == "__main__":
    # <<<<<<<<<ITER VQE PARAMETERS>>>>>>>>>>>>>>>>>>>>

    # <<<<<<<<<<< MOLECULE PARAMETERS >>>>>>>>>>>>>
    r = 1.5
    # theta = 0.538*numpy.pi # for H20
    frozen_els = {'occupied': [], 'unoccupied': []}
    molecule = H6(r=r)  # (frozen_els=frozen_els)

    # <<<<<<<<<< ANSATZ ELEMENT POOL PARAMETERS >>>>>>>>>>>>.
    ansatz_element_type = 'eff_f_exc'
    # ansatz_element_type = 'q_exc'
    # ansatz_element_type = 'f_exc'
    # ansatz_element_type = 'pauli_str_exc'
    q_encoding = 'jw'
    spin_complement = False  # only for fermionic and qubit excitations (not for PWEs)

    # <<<<<<<<<< TERMINATION PARAMETERS >>>>>>>>>>>>>>>>>
    delta_e_threshold = 1e-12  # 1e-3 for chemical accuracy
    max_ansatz_elements = 600

    # <<<<<<<<<<<< DEFINE BACKEND >>>>>>>>>>>>>>>>>
    backend = backends.MatrixCacheBackend

    # <<<<<<<<<< DEFINE OPTIMIZER >>>>>>>>>>>>>>>>>
    use_energy_vector_gradient = True  # for optimizer

    # create a vqe_runner object
    vqe_runner = VQERunner(molecule, backend=backend, optimizer='BFGS', optimizer_options={'gtol': 1e-08},
                           use_ansatz_gradient=use_energy_vector_gradient)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    LogUtils.log_config()
    logging.info('{}, r={} ,{}'.format(molecule.name, r, ansatz_element_type))
    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

    # create the pool of ansatz elements
    if spin_complement:
        ansatz_element_pool = SpinCompGSDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                                     element_type=ansatz_element_type, encoding=q_encoding).get_excitations()
    else:
        ansatz_element_pool = GSDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                            ansatz_element_type=ansatz_element_type).get_excitations()

    # create simulation cache
    if backend == backends.MatrixCacheBackend:
        # precompute commutator matrices, that are use in excitation gradient calculation
        global_cache = GlobalCache(molecule)
        global_cache.calculate_exc_gen_sparse_matrices_dict(ansatz_element_pool)
        global_cache.calculate_commutators_sparse_matrices_dict(ansatz_element_pool)
    else:
        global_cache = None

    message = 'Length of new pool', len(ansatz_element_pool)
    logging.info(message)

    # initialize a dataFrame to collect the simulation data
    results_data_frame = pandas.DataFrame(columns=['n', 'E', 'dE', 'error', 'n_iters', 'cnot_count', 'u1_count',
                                                   'cnot_depth', 'u1_depth', 'element', 'element_qubits',
                                                   'var_parameters'])
    # <<<<<<<<<<<< LOAD PAUSED SIMULATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    init_db = pandas.read_csv("../../results/iter_vqe_results/H6_iqeb_q_exc_n=1_r=15_no_comps_02-June-2021.csv")

    if init_db is None:
        ansatz_elements = []
        ansatz_parameters = []
    else:
        state = DataUtils.ansatz_from_data_frame(init_db, molecule)
        ansatz_elements = state.ansatz_elements
        ansatz_parameters = state.parameters

    # <<<<<<<<<<<<<< INITIALIZE ITERATIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # ansatz_elements = []
    # ansatz_parameters = []

    iter_count = 0
    df_count = 0
    exact_energy = molecule.fci_energy
    print('Exact energy ', exact_energy)
    current_energy = vqe_runner.backend.ham_expectation_value([], [], molecule, global_cache)

    print(current_energy)
    previous_energy = current_energy + max(delta_e_threshold, 1e-5)
    init_ansatz_length = len(ansatz_elements)

    while previous_energy - current_energy >= delta_e_threshold and iter_count <= max_ansatz_elements:
        iter_count += 1

        print('New iteration ', iter_count)

        previous_energy = current_energy

        element_to_add, grad = GradientUtils.\
            get_largest_gradient_elements(ansatz_element_pool, molecule, backend=vqe_runner.backend,
                                          ansatz_parameters=ansatz_parameters, ansatz=ansatz_elements,
                                          global_cache=global_cache)[0]
        print(element_to_add.element)

        result = vqe_runner.vqe_run(ansatz=ansatz_elements + [element_to_add], init_guess_parameters=ansatz_parameters + [0],
                                    cache=global_cache)

        current_energy = result.fun
        delta_e = previous_energy - current_energy

        # get initial guess for the var. params. for the next iteration
        ansatz_parameters = list(result.x)

        if delta_e > 0:

            ansatz_elements.append(element_to_add)

            # save iteration data
            element_qubits = element_to_add.qubits

            gate_count = IterVQEQasmUtils.gate_count_from_ansatz(ansatz_elements, molecule.n_orbitals)
            results_data_frame.loc[iter_count] = {'n': iter_count, 'E': current_energy, 'dE': delta_e,
                                                  'error': current_energy - exact_energy, 'n_iters': result['n_iters'],
                                                  'cnot_count': gate_count['cnot_count'], 'u1_count': gate_count['u1_count'],
                                                  'cnot_depth': gate_count['cnot_depth'], 'u1_depth': gate_count['u1_depth'],
                                                  'element': element_to_add.element, 'element_qubits': element_qubits,
                                                  'var_parameters': 0}
            results_data_frame['var_parameters'] = list(result.x)[init_ansatz_length:]
            # df_data['var_parameters'] = var_parameters
            # save data
            DataUtils.save_data(results_data_frame, molecule, time_stamp, frozen_els=frozen_els,
                                ansatz_element_type=ansatz_element_type, iter_vqe_type='adapt')

            message = 'Add new element to final ansatz {}. Energy {}. Energy change {}, Grad{}, var. parameters: {}' \
                .format(element_to_add.element, current_energy, delta_e, grad, ansatz_parameters)
            logging.info(message)
        else:
            message = 'No contribution to energy decrease. Stop adding elements to the final ansatz'
            logging.info(message)
            break

        print('Added element ', ansatz_elements[-1].element)

    # calculate the VQE for the final ansatz
    final_result = vqe_runner.vqe_run(ansatz=ansatz_elements, cache=global_cache)
    t = time.time()

    print(final_result)
    print('Ciao')
