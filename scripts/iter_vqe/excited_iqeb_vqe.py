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
from src.molecules.molecules import *
from src.ansatz_element_sets import *
from src.backends import QiskitSimBackend
from src.utils import LogUtils
from src.iter_vqe_utils import *
from src.cache import *


if __name__ == "__main__":
    # <<<<<<<<<ITER VQE PARAMETERS>>>>>>>>>>>>>>>>>>>>

    # <<<<<<<<<<< MOLECULE PARAMETERS >>>>>>>>>>>>>
    r = 1
    # theta = 0.538*numpy.pi # for H20
    frozen_els = {'occupied': [], 'unoccupied': []}
    molecule = LiH(r=r)  # (frozen_els=frozen_els)
    excited_state = 1

    # <<<<<<<<<<,get lower energy states>>>>>>>>>>>>>
    # molecule.default_states()
    df = pandas.read_csv('../../results/iter_vqe_results/vip/LiH_h_adapt_gsdqe_comp_pairs_r=1_24-Sep-2020.csv')
    # df = pandas.read_csv('../../results/iter_vqe_results/vip/LiH_h_adapt_gsdqe_comp_pair_r=3_24-Sep-2020.csv')
    # df = pandas.read_csv('../../results/iter_vqe_results/LiH_iqeb_q_exc_r=1.25_19-Nov-2020.csv')

    ground_state = DataUtils.ansatz_from_data_frame(df, molecule)
    molecule.H_lower_state_terms = [[abs(molecule.hf_energy)*2, ground_state]]

    n_largest_grads = 10
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

    # vqe_runner_2 = VQERunner(molecule, backend=backend, optimizer='BFGS', optimizer_options={'gtol': 1e-08},
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
    results_data_frame = pandas.DataFrame(columns=['n', 'E', 'dE', 'rank', 'error', 'n_iters', 'cnot_count', 'u1_count',
                                          'cnot_depth', 'u1_depth', 'element', 'element_qubits', 'var_parameters'])

    # <<<<<<<<<<<<<< INITIALIZE ITERATIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ansatz = []
    ansatz_parameters = []

    iter_count = 0
    df_count = 0
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

        elements_energies = EnergyUtils.\
            largest_individual_vqe_energy_reduction_elements(vqe_runner_2, ansatz_element_pool, ansatz=ansatz,
                                                             ansatz_parameters=ansatz_parameters, excited_state=excited_state,
                                                             n=n_largest_grads, global_cache=global_cache)
        elements = [e_g[0] for e_g in elements_energies]
        elements_keys = [str(el.excitations_generators) for el in elements]
        elements_parameters = [e_g[1].x[0] for e_g in elements_energies]
        dEs = [e_g[1].fun - current_energy for e_g in elements_energies]

        message = 'Elements with largest individual energy reductions {}. dEs {}'.format([el.element for el in elements], dEs)
        logging.info(message)

        element_to_add, intermediate_result = \
            EnergyUtils.largest_full_vqe_energy_reduction_element(vqe_runner, elements, elements_parameters=elements_parameters,
                                                                  ansatz=ansatz, ansatz_parameters=ansatz_parameters,
                                                                  global_cache=global_cache,
                                                                  excited_state=excited_state)

        el_to_add_rank = elements_keys.index(str(element_to_add.excitations_generators))

        ansatz.append(element_to_add)
        new_ansatz_elements = [element_to_add]
        ansatz_parameters = list(intermediate_result.x)
        result = intermediate_result
        current_energy = intermediate_result.fun
        delta_e = previous_energy - current_energy

        if delta_e > 0:
            for new_ansatz_element in new_ansatz_elements:
                # save iteration data
                element_qubits = new_ansatz_element.qubits

                gate_count = IterVQEQasmUtils.gate_count_from_ansatz(ansatz, molecule.n_orbitals)
                results_data_frame.loc[df_count] = {'n': iter_count, 'E': current_energy, 'dE': delta_e,
                                                    'error': current_energy - exact_energy,
                                                    'rank': el_to_add_rank, 'n_iters': result['n_iters'],
                                                    'cnot_count': gate_count['cnot_count'],
                                                    'u1_count': gate_count['u1_count'],
                                                    'cnot_depth': gate_count['cnot_depth'],
                                                    'u1_depth': gate_count['u1_depth'],
                                                    'element': new_ansatz_element.element,
                                                    'element_qubits': element_qubits, 'var_parameters': 0}
                df_count += 1
                results_data_frame['var_parameters'] = list(result.x)[:df_count]

                # save data
                DataUtils.save_data(results_data_frame, molecule, time_stamp, ansatz_element_type=ansatz_element_type,
                                    frozen_els=frozen_els, iter_vqe_type='exc_iqeb')

                message = 'Add new element to final ansatz {}. Energy {}. Energy change {}, var. parameters: {}' \
                    .format(element_to_add.element, current_energy, delta_e, ansatz_parameters)
                logging.info(message)
        else:
            message = 'No contribution to energy decrease. Stop adding elements to the final ansatz'
            logging.info(message)
            break

        print('Added element ', ansatz[-1].element)

    # calculate the VQE for the final ansatz
    final_result = vqe_runner.vqe_run(ansatz=ansatz, cache=global_cache, excited_state=excited_state)
    t = time.time()

    print(final_result)
    print('Ciao')
