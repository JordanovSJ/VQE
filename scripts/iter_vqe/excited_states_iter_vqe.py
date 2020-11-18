import logging
import time
import numpy
import pandas
import ray
import datetime
import sys
import ast
sys.path.append('../../')

from src.vqe_runner import VQERunner
from src.q_systems import *
from src.ansatz_element_sets import *
from src.backends import QiskitSimBackend
from src.utils import LogUtils
from src.iter_vqe_utils import *
from src.cache import *


if __name__ == "__main__":
    # <<<<<<<<<ITER VQE PARAMETERS>>>>>>>>>>>>>>>>>>>>

    # <<<<<<<<<<< MOLECULE PARAMETERS >>>>>>>>>>>>>
    r = 1.546
    # theta = 0.538*numpy.pi # for H20
    frozen_els = {'occupied': [], 'unoccupied': []}
    molecule = LiH(r=r)  # (frozen_els=frozen_els)
    excited_state = 1
    molecule.default_states()

    # <<<<<<<<<< ANSATZ ELEMENT POOL PARAMETERS >>>>>>>>>>>>.
    # ansatz_element_type = 'eff_f_exc'
    ansatz_element_type = 'q_exc'
    # ansatz_element_type = 'pauli_str_exc'
    spin_complement = False  # only for fermionic and qubit excitations (not for PWEs)

    # <<<<<<<<<< TERMINATION PARAMETERS >>>>>>>>>>>>>>>>>
    delta_e_threshold = 1e-12  # 1e-3 for chemical accuracy
    max_ansatz_elements = 250

    # <<<<<<<<<<<< DEFINE BACKEND >>>>>>>>>>>>>>>>>
    backend = backends.MatrixCacheBackend

    # <<<<<<<<<< DEFINE OPTIMIZER >>>>>>>>>>>>>>>>>
    use_energy_vector_gradient = True  # for optimizer

    # create a vqe_runner object
    vqe_runner = VQERunner(molecule, backend=backend, optimizer='BFGS', optimizer_options={'gtol': 1e-08},
                           use_ansatz_gradient=use_energy_vector_gradient)

    # create a vqe_runner for excited states, where the minimum may be away from the zero, which will make gradient
    # descent optimizers useless

    # vqe_runner_2 = VQERunner(molecule, backend=QiskitSim, optimizer='BFGS', optimizer_options={'gtol': 1e-08},
    #                          use_ansatz_gradient=use_grad)
    vqe_runner_2 = VQERunner(molecule, backend=backend, optimizer='Nelder-Mead', optimizer_options=None)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    LogUtils.log_config()
    logging.info('{}, r={} ,{}'.format(molecule.name, r, ansatz_element_type))
    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

    # create the pool of ansatz elements
    if spin_complement:
        ansatz_element_pool = SpinCompGSDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                                     element_type=ansatz_element_type).get_excitations()
    else:
        ansatz_element_pool = GSDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                             ansatz_element_type=ansatz_element_type).get_excitations()

    message = 'Length of new pool', len(ansatz_element_pool)
    logging.info(message)

    # create simulation cache
    if backend == backends.MatrixCacheBackend:
        # precompute commutator matrices, that are use in excitation gradient calculation
        global_cache = GlobalCache(molecule, excited_state=excited_state)
        global_cache.calculate_exc_gen_sparse_matrices_dict(ansatz_element_pool)
        global_cache.calculate_commutators_sparse_matrices_dict(ansatz_element_pool)
    else:
        global_cache = None

    # initialize a dataFrame to collect the simulation data
    results_data_frame = pandas.DataFrame(columns=['n', 'E', 'dE', 'error', 'n_iters', 'cnot_count', 'u1_count',
                                          'cnot_depth', 'u1_depth', 'element', 'element_qubits', 'var_parameters'])

    # <<<<<<<<<<<<<< INITIALIZE ITERATIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ansatz = []
    var_parameters = []

    iter_count = 0
    exact_energy = molecule.calculate_energy_eigenvalues(excited_state+1)[excited_state]
    print('Exact energy ', exact_energy)
    current_energy = vqe_runner.backend.ham_expectation_value([], [], molecule, cache=global_cache, excited_state=excited_state)

    print(current_energy)
    previous_energy = current_energy + max(delta_e_threshold, 1e-5)
    init_ansatz_length = len(ansatz)

    while previous_energy - current_energy >= delta_e_threshold and iter_count <= max_ansatz_elements:
        iter_count += 1

        print('New iteration ', iter_count)

        previous_energy = current_energy

        element_to_add, element_result = EnergyUtils.\
            largest_individual_vqe_energy_reduction_elements(vqe_runner_2, ansatz_element_pool, ansatz=ansatz,
                                                             ansatz_parameters=var_parameters, excited_state=excited_state,
                                                             global_cache=global_cache)
        element_energy_reduction = element_result.fun
        print(element_to_add.element)

        result = vqe_runner.vqe_run(ansatz=ansatz + [element_to_add],
                                    init_guess_parameters=var_parameters + list(element_result.x),
                                    excited_state=excited_state, cache=global_cache)

        current_energy = result.fun
        delta_e = previous_energy - current_energy
        var_parameters = list(result.x)

        if delta_e > 0:
            ansatz.append(element_to_add)

            # save iteration data
            try:
                element_qubits = element_to_add.qubits
            except AttributeError:
                element_qubits = []

            gate_count = IterVQEQasmUtils.gate_count_from_ansatz(ansatz, molecule.n_orbitals)

            results_data_frame.loc[iter_count] = {'n': iter_count, 'E': current_energy, 'dE': delta_e,
                                                  'error': current_energy - exact_energy, 'n_iters': result['n_iters'],
                                                  'cnot_count': gate_count['cnot_count'], 'u1_count': gate_count['u1_count'],
                                                  'cnot_depth': gate_count['cnot_depth'], 'u1_depth': gate_count['u1_depth'],
                                                  'element': element_to_add.element, 'element_qubits': element_qubits,
                                                  'var_parameters': 0}
            results_data_frame['var_parameters'] = var_parameters
            DataUtils.save_data(results_data_frame, molecule, time_stamp, frozen_els=frozen_els,
                                ansatz_element_type=ansatz_element_type, iter_vqe_type='exc_{}_iter_vqe'.format(excited_state))

            message = 'Add new element to the ansatz {}. Energy {}. dE {}, Individual dE{}, var. parameters: {}' \
                .format(element_to_add.element, current_energy, delta_e, element_energy_reduction, var_parameters)
            logging.info(message)
        else:
            message = 'No contribution to energy decrease. Stop adding elements to the final ansatz'
            logging.info(message)
            break

    # calculate the VQE for the final ansatz
    final_result = vqe_runner_2.vqe_run(ansatz=ansatz, cache=global_cache, excited_state=excited_state)
    t = time.time()

    print(final_result)
    print('Ciao')
