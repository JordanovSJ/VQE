import pandas

import sys
sys.path.append('../../')

from src.vqe_runner import VQERunner
from src.q_systems import *
from src.ansatz_element_sets import *
from src.backends import QiskitSim
from src.utils import *
from src.iter_vqe_utils import *
from src.cache import *


if __name__ == "__main__":
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<,simulation parameters>>>>>>>>>>>>>>>>>>>>
    r = 0.735
    # theta = 0.538*numpy.pi # for H20
    frozen_els = {'occupied': [], 'unoccupied': []}
    molecule = H4() #(frozen_els=frozen_els)

    backend = QiskitSim

    ansatz_element_type = 'q_exc'

    delta_e_threshold = 1e-12
    max_ansatz_size = 90

    use_grad = True  # for optimizer
    # size_patch_commutators = 500  # not used

    n_largest_grads = 20

    init_ansatz_df = None #pandas.read_csv("../../results/adapt_vqe_results/vip/LiH_h_adapt_gsdqe_27-Jul-2020.csv")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    LogUtils.log_config()
    logging.info('{}, r={} ,{}'.format(molecule.name, r, ansatz_element_type))

    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

    # create a vqe runner object
    optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-08}
    vqe_runner = VQERunner(molecule, backend=backend, optimizer=optimizer, optimizer_options=optimizer_options,
                           use_ansatz_gradient=use_grad)

    # dataFrame to collect the simulation data
    results_data_frame = pandas.DataFrame(columns=['n', 'E', 'dE', 'error', 'n_iters', 'cnot_count', 'u1_count', 'cnot_depth',
                                         'u1_depth', 'element', 'element_qubits', 'var_parameters'])

    ansatz_element_pool = GSDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                         ansatz_element_type=ansatz_element_type).get_excitations()
    print('Pool len: ', len(ansatz_element_pool))

    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>?>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if config.use_cache:
        # precompute commutator matrices, that are use in excitation gradient calculation
        global_cache = GlobalCache(molecule)  # excited_state=excited_state)
        global_cache.calculate_exc_gen_sparse_matrices_list(ansatz_element_pool)
        global_cache.calculate_commutators_matrices(ansatz_element_pool)
    else:
        global_cache = None

    # load an old simulation ansatz
    if init_ansatz_df is None:
        ansatz = []
        var_parameters = []
    else:
        ansatz, var_parameters = DataUtils.ansatz_from_data_frame(init_ansatz_df, molecule)
        # var_parameters = list(vqe_runner.vqe_run(ansatz_elements, var_parameters).x)

    hf_energy = molecule.hf_energy
    fci_energy = molecule.fci_energy
    iter_count = 0
    df_count = 0
    current_energy = hf_energy
    previous_energy = 0
    init_ansatz_length = len(ansatz)

    while previous_energy - current_energy >= delta_e_threshold and iter_count <= max_ansatz_size:
        iter_count += 1

        print('New cycle ', iter_count)

        previous_energy = current_energy

        # get the n elements with largest gradients
        elements_grads = GradientUtils.\
            get_largest_gradient_elements(ansatz_element_pool, molecule, backend=backend, n=n_largest_grads,
                                          ansatz_parameters=var_parameters, ansatz=ansatz, global_cache=global_cache)

        elements = [e_g[0] for e_g in elements_grads]
        grads = [e_g[1] for e_g in elements_grads]

        message = 'Elements with largest grads {}. Grads {}'.format([el.element for el in elements], grads)
        logging.info(message)

        element_to_add, result =\
            EnergyUtils.largest_full_vqe_energy_reduction_element(vqe_runner, elements, ansatz_parameters=var_parameters,
                                                                  ansatz=ansatz, global_cache=global_cache)

        compl_element_to_add = element_to_add.get_spin_comp_exc()

        # TODO check the if condition
        comp_qubits = compl_element_to_add.qubits
        qubits = element_to_add.qubits
        if [set(qubits[0]), set(qubits[1])] == [set(comp_qubits[0]), set(comp_qubits[1])] or \
           [set(qubits[0]), set(qubits[1])] == [set(comp_qubits[1]), set(comp_qubits[0])]:
            result = vqe_runner.vqe_run(ansatz=ansatz + [element_to_add],
                                        init_guess_parameters=var_parameters + [0], cache=global_cache)
            current_energy = result.fun
            delta_e = previous_energy - current_energy
            if delta_e > 0:
                ansatz.append(element_to_add)
                new_ansatz_elements = [element_to_add]
        else:
            result = vqe_runner.vqe_run(ansatz=ansatz + [element_to_add, compl_element_to_add],
                                        init_guess_parameters=var_parameters + [0, 0], cache=global_cache)
            current_energy = result.fun
            delta_e = previous_energy - current_energy
            if delta_e > 0:
                ansatz.append(element_to_add)
                ansatz.append(compl_element_to_add)
                print('Add complement element: ', compl_element_to_add.element)
                new_ansatz_elements = [element_to_add, compl_element_to_add]

        # get initial guess for the var. params. for the next iteration
        var_parameters = list(result.x)

        if delta_e > 0:
            for new_ansatz_element in new_ansatz_elements:
                # save iteration data
                element_qubits = new_ansatz_element.qubits

                gate_count = IterVQEQasmUtils.gate_count_from_ansatz(ansatz, molecule.n_orbitals)
                results_data_frame.loc[df_count] = {'n': iter_count, 'E': current_energy, 'dE': delta_e,
                                         'error': current_energy - fci_energy,
                                         'n_iters': result['n_iters'], 'cnot_count': gate_count['cnot_count'],
                                         'u1_count': gate_count['u1_count'], 'cnot_depth': gate_count['cnot_depth'],
                                         'u1_depth': gate_count['u1_depth'], 'element': new_ansatz_element.element,
                                         'element_qubits': element_qubits, 'var_parameters': 0}
                df_count += 1
                results_data_frame['var_parameters'] = list(result.x)[:df_count]
                # df_data['var_parameters'] = var_parameters

                # save data
                DataUtils.save_data(results_data_frame, molecule, time_stamp, ansatz_element_type=ansatz_element_type,
                                    frozen_els=frozen_els, iter_vqe_type='iqeb')

                message = 'Add new element to final ansatz {}. Energy {}. Energy change {}, var. parameters: {}' \
                    .format(element_to_add.element, current_energy, delta_e, var_parameters)
                logging.info(message)
        else:
            message = 'No contribution to energy decrease. Stop adding elements to the final ansatz'
            logging.info(message)
            break

        print('Added element ', ansatz[-1].element)

    # calculate the VQE for the final ansatz
    final_result = vqe_runner.vqe_run(ansatz=ansatz)
    t = time.time()

    print(final_result)
    print('Ciao')