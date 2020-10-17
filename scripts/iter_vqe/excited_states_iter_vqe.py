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
from src.ansatze import *
from src.backends import QiskitSim
from src.utils import LogUtils
from src.iter_vqe_utils import *


if __name__ == "__main__":
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<,simulation parameters>>>>>>>>>>>>>>>>>>>>
    r = 1.546
    # theta = 0.538*numpy.pi # for H20
    frozen_els = {'occupied': [], 'unoccupied': []}
    molecule = LiH()  # (frozen_els=frozen_els)
    excited_state = 1
    molecule.default_states()

    # ansatz_element_type = 'eff_f_exc'
    ansatz_element_type = 'q_exc'
    # ansatz_element_type = 'pauli_str_exc'
    spin_complement = False  # only for fermionic and qubit excitations (not for PWEs)

    delta_e_threshold = 1e-12  # 1e-3 for chemical accuracy
    max_ansatz_elements = 250

    multithread = True
    use_grad = True  # for optimizer
    use_commutators_cache = True
    use_backend_cache = True
    # size_patch_commutators = 500  # not used

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    LogUtils.log_config()
    logging.info('{}, r={} ,{}'.format(molecule.name, r, ansatz_element_type))

    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

    # create a vqe_runner object
    vqe_runner = VQERunner(molecule, backend=QiskitSim, optimizer='BFGS', optimizer_options={'gtol': 1e-08},
                           use_ansatz_gradient=use_grad, use_cache=use_backend_cache)

    # create a vqe_runner for single element global optimization
    vqe_runner_2 = VQERunner(molecule, backend=QiskitSim, optimizer='BFGS', optimizer_options={'gtol': 1e-08},
                           use_ansatz_gradient=use_grad, use_cache=use_backend_cache)
    # vqe_runner_2 = VQERunner(molecule, backend=QiskitSim, optimizer='Nelder-Mead', use_cache=use_backend_cache,
    #                          optimizer_options=None)

    # define the pool of ansatz elements
    if spin_complement:
        ansatz_element_pool = SpinCompGSDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                                     element_type=ansatz_element_type).get_excitations()
    else:
        ansatz_element_pool = GSDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                             ansatz_element_type=ansatz_element_type).get_excitations()
    message = 'Length of new pool', len(ansatz_element_pool)
    logging.info(message)
    print(message)

    # TODO excited state commutators
    # precompute commutator matrices, that are use in excitation gradient calculation
    if use_commutators_cache:
        commutators_cache = GradUtils.calculate_commutators(ansatz_elements=ansatz_element_pool,
                                                            q_system=molecule, multithread=multithread,
                                                            excited_state=excited_state)
    else:
        commutators_cache = None

    # initialize a dataFrame to collect the simulation data
    results_data_frame = pandas.DataFrame(columns=['n', 'E', 'dE', 'error', 'n_iters', 'cnot_count', 'u1_count',
                                          'cnot_depth', 'u1_depth', 'element', 'element_qubits', 'var_parameters'])
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>?>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ansatz = []
    var_parameters = []

    iter_count = 0
    exact_energy = molecule.calculate_energy_eigenvalues(excited_state+1)[excited_state]
    print(exact_energy)
    current_energy = vqe_runner.backend.ham_expectation_value_exc_state(molecule.jw_qubit_ham, [], [], molecule,
                                                                        excited_state=excited_state)
    print(current_energy)
    previous_energy = current_energy + max(delta_e_threshold,1e-5)
    init_ansatz_length = len(ansatz)

    while previous_energy - current_energy >= delta_e_threshold and iter_count <= max_ansatz_elements:
        iter_count += 1

        print('New iteration ', iter_count)

        previous_energy = current_energy

        element_to_add, element_result = EnergyUtils.\
            largest_individual_vqe_energy_reduction_element(ansatz_element_pool, vqe_runner_2, ansatz=ansatz,
                                                            var_parameters=var_parameters, multithread=multithread,
                                                            excited_state=excited_state, commutators_cache=commutators_cache)
        element_energy_reduction = element_result.fun
        print(element_to_add.element)

        result = vqe_runner.vqe_run(ansatz=ansatz + [element_to_add],
                                    initial_var_parameters=var_parameters + list(element_result.x), excited_state=excited_state)

        current_energy = result.fun
        delta_e = previous_energy - current_energy

        # get initial guess for the var. params. for the next iteration
        var_parameters = list(result.x)

        if delta_e > 0:

            ansatz.append(element_to_add)

            # save iteration data
            element_qubits = element_to_add.qubits

            gate_count = IterVQEQasmUtils.gate_count_from_ansatz(ansatz, molecule.n_orbitals)
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

            message = 'Add new element to final ansatz {}. Energy {}. dE {}, Individual dE{}, var. parameters: {}' \
                .format(element_to_add.element, current_energy, delta_e, element_energy_reduction, var_parameters)
            logging.info(message)
            print(message)
        else:
            message = 'No contribution to energy decrease. Stop adding elements to the final ansatz'
            logging.info(message)
            print(message)
            break

        print('Added element ', ansatz[-1].element)

    # calculate the VQE for the final ansatz
    final_result = vqe_runner.vqe_run(ansatz=ansatz, excited_state=excited_state)
    t = time.time()

    print(final_result)
    print('Ciao')
