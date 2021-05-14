import logging
import time
import numpy
import pandas
import ray
import pandas
import datetime

import sys
import ast
sys.path.append('../')

from src.vqe_runner import VQERunner
from src.molecular_system import *
from src.ansatz_element_lists import *
from src.backends import QiskitSimBackend
from src.utils import LogUtils
from src.adapt_utils import *


def save_data(df_data, molecule, time_stamp, ansatz_element_type=None, frozen_els=None):
    if ansatz_element_type is None:
        ansatz_element_type = 'unspecified'

    try:
        df_data.to_csv('../../results/adapt_vqe_results/{}_{}_{}_{}.csv'.format(molecule.name, ansatz_element_type, frozen_els, time_stamp))
    except FileNotFoundError:
        try:
            df_data.to_csv('results/adapt_vqe_results/{}_{}_{}_{}.csv'.format(molecule.name, ansatz_element_type, frozen_els, time_stamp))
        except FileNotFoundError as fnf:
            print(fnf)


def get_ansatz_from_csv(db, molecule):
    ansatz = []
    for i in range(len(db)):
        element = db.loc[i]['element']
        element_qubits = db.loc[i]['element_qubits']
        if element[0] == 'e' and element[4] == 's':
            ansatz.append(EffSFExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        elif element[0] == 'e' and element[4] == 'd':
            ansatz.append(EffDFExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        elif element[0] == 's' and element[2] == 'q':
            ansatz.append(SQExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        elif element[0] == 'd' and element[2] == 'q':
            ansatz.append(DQExc(*ast.literal_eval(element_qubits), system_n_qubits=molecule.n_qubits))
        else:
            print(element, element_qubits)
            raise Exception('Unrecognized ansatz element.')

    var_pars = list(db['var_parameters'])

    return ansatz, var_pars


def get_spin_compl_exc(qubits_1, qubits_2, element_type, n_qubits):

    if element_type == 'qubit_excitation':
        if len(qubits_1) == 1 and len(qubits_2) == 1:
            return SQExc(AnsatzElement.spin_complement_orbital(qubits_1[0]),
                         AnsatzElement.spin_complement_orbital(qubits_2[0]), system_n_qubits=n_qubits)
        elif len(qubits_1) == 2 and len(qubits_2) == 2:
            return DQExc([AnsatzElement.spin_complement_orbital(qubits_1[0]),
                          AnsatzElement.spin_complement_orbital(qubits_1[1])],
                         [AnsatzElement.spin_complement_orbital(qubits_2[0]),
                          AnsatzElement.spin_complement_orbital(qubits_2[1])], system_n_qubits=n_qubits)
        else:
            raise Exception('Wrong qubits format')
    elif element_type == 'fermionic_excitation':
        if len(qubits_1) == 1 and len(qubits_2) == 1:
            return EffSFExc(AnsatzElement.spin_complement_orbital(qubits_1[0]),
                            AnsatzElement.spin_complement_orbital(qubits_2[0]), system_n_qubits=n_qubits)
        elif len(qubits_1) == 2 and len(qubits_2) == 2:
            return EffDFExc([AnsatzElement.spin_complement_orbital(qubits_1[0]),
                             AnsatzElement.spin_complement_orbital(qubits_1[1])],
                            [AnsatzElement.spin_complement_orbital(qubits_2[0]),
                             AnsatzElement.spin_complement_orbital(qubits_2[1])], system_n_qubits=n_qubits)
        else:
            raise Exception('Wrong qubits format')
    else:
        raise Exception('Wrong element type')


if __name__ == "__main__":
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<,simulation parameters>>>>>>>>>>>>>>>>>>>>
    r = 0.735
    # theta = 0.538*numpy.pi # for H20
    frozen_els = {'occupied': [], 'unoccupied': []}
    molecule = H4() #(frozen_els=frozen_els)

    # ansatz_element_type = 'efficient_fermi_excitation'
    ansatz_element_type = 'qubit_excitation'
    ## ansatz_element_type = 'pauli_word_excitation'

    accuracy = 1e-12  # 1e-3 for chemical accuracy
    # threshold = 1e-14
    max_ansatz_size = 90

    multithread = True
    use_grad = True  # for optimizer
    do_precompute_commutators = True
    size_patch_commutators = 5

    do_precompute_statevector = True  # for ansatz elements grad computation

    n_largest_grads = 20

    init_db = None #pandas.read_csv("../../results/adapt_vqe_results/vip/LiH_h_adapt_gsdqe_27-Jul-2020.csv")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    LogUtils.log_config()
    logging.info('{}, r={} ,{}'.format(molecule.name, r, ansatz_element_type))

    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

    # create a vqe runner object
    optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-08}
    vqe_runner = VQERunner(molecule, backend=QiskitSimBackend, optimizer=optimizer, optimizer_options=optimizer_options,
                           use_ansatz_gradient=use_grad)
    hf_energy = molecule.hf_energy
    fci_energy = molecule.fci_energy

    # dataFrame to collect the simulation data
    df_data = pandas.DataFrame(columns=['n', 'E', 'dE', 'error', 'n_iters', 'cnot_count', 'u1_count', 'cnot_depth',
                                        'u1_depth', 'element', 'element_qubits', 'var_parameters'])

    ansatz_element_pool = GSDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                         ansatz_element_type=ansatz_element_type).get_excitations()
    print('Pool len: ', len(ansatz_element_pool))

    if do_precompute_commutators:
        dynamic_commutators = {}
        print('Calculating commutators')
        for i in range(int(len(ansatz_element_pool) / size_patch_commutators)):
            patch_ansatz_elements = ansatz_element_pool[i * size_patch_commutators:(i + 1) * size_patch_commutators]
            patch_dynamic_commutators = GradAdaptUtils.calculate_commutators(H_qubit_operator=molecule.qubit_ham,
                                                                             ansatz_elements=patch_ansatz_elements,
                                                                             n_system_qubits=molecule.n_orbitals,
                                                                             multithread=multithread)
            print(i)
            dynamic_commutators = {**dynamic_commutators, **patch_dynamic_commutators}
            mem_size = 0
            for x in dynamic_commutators:
                mem_size += dynamic_commutators[x].data.nbytes
                print('Commutators size ', mem_size)
            del patch_dynamic_commutators
            del patch_ansatz_elements

        patch_ansatz_elements = ansatz_element_pool[
                                int(len(ansatz_element_pool) / size_patch_commutators) * size_patch_commutators:]
        patch_dynamic_commutators = GradAdaptUtils.calculate_commutators(H_qubit_operator=molecule.qubit_ham,
                                                                         ansatz_elements=patch_ansatz_elements,
                                                                         n_system_qubits=molecule.n_orbitals,
                                                                         multithread=multithread)
        dynamic_commutators = {**dynamic_commutators, **patch_dynamic_commutators}
        del patch_dynamic_commutators
        del patch_ansatz_elements
        print('Finished calculating commutators')
    else:
        dynamic_commutators = None

    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>?>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    message = 'Length of new pool', len(ansatz_element_pool)
    logging.info(message)
    print(message)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    if init_db is None:
        ansatz_elements = []
        var_parameters = []
    else:
        ansatz_elements, var_parameters = get_ansatz_from_csv(init_db, molecule)

    # var_parameters = list(vqe_runner.vqe_run(ansatz_elements, var_parameters).x)

    iter_count = 0
    df_count = 0
    current_energy = hf_energy
    previous_energy = 0
    init_ansatz_length = len(ansatz_elements)

    while previous_energy - current_energy >= accuracy and iter_count <= max_ansatz_size:
        iter_count += 1

        print('New cycle ', iter_count)

        previous_energy = current_energy

        # get the n elements with largest gradients
        elements_grads = GradAdaptUtils.\
            get_largest_gradient_elements(ansatz_element_pool, molecule, vqe_runner.backend, n=n_largest_grads,
                                          ansatz_parameters=var_parameters, ansatz=ansatz_elements,
                                          multithread=multithread, do_precompute_statevector=do_precompute_statevector,
                                          commutators_cache=dynamic_commutators)

        elements = [e_g[0] for e_g in elements_grads]
        grads = [e_g[1] for e_g in elements_grads]

        message = 'Elements with largest grads {}. Grads {}'.format([el.element for el in elements], grads)
        logging.info(message)
        print(message)

        element_to_add, result =\
            EnergyAdaptUtils.largest_individual_vqe_energy_reduction_elements(vqe_runner, elements,
                                                                              ansatz_parameters=var_parameters,
                                                                              ansatz=ansatz_elements, multithread=multithread)

        compl_element_to_add = None
        if element_to_add.order == 1:
            # element_qubits = [element_to_add.qubit_1, element_to_add.qubit_2]
            if {element_to_add.qubit_1, element_to_add.qubit_2} \
                    != {AnsatzElement.spin_complement_orbital(element_to_add.qubit_1),
                        AnsatzElement.spin_complement_orbital(element_to_add.qubit_2)}:

                compl_element_to_add = get_spin_compl_exc([element_to_add.qubit_1], [element_to_add.qubit_2],
                                                          ansatz_element_type, molecule.n_orbitals)
                print('Add complement element: ', compl_element_to_add.element)

        elif element_to_add.order == 2:
            # element_qubits = [element_to_add.qubit_pair_1, element_to_add.qubit_pair_2]
            qubit_pair_1 = element_to_add.qubit_pair_1
            comp_qubit_pair_1 = [AnsatzElement.spin_complement_orbital(qubit_pair_1[0]),
                                 AnsatzElement.spin_complement_orbital(qubit_pair_1[1])]

            qubit_pair_2 = element_to_add.qubit_pair_2
            comp_qubit_pair_2 = [AnsatzElement.spin_complement_orbital(qubit_pair_2[0]),
                                 AnsatzElement.spin_complement_orbital(qubit_pair_2[1])]

            if set(qubit_pair_1) != set(comp_qubit_pair_1) or set(qubit_pair_2) != set(comp_qubit_pair_2):
                compl_element_to_add = get_spin_compl_exc(element_to_add.qubit_pair_1, element_to_add.qubit_pair_2,
                                                          ansatz_element_type, molecule.n_orbitals)
                print('Add complement element: ', compl_element_to_add.element)
        else:
            raise Exception('Wrong element order ..')

        if compl_element_to_add is None:
            result = vqe_runner.vqe_run(ansatz=ansatz_elements + [element_to_add],
                                        init_guess_parameters=var_parameters + [0])
        else:
            result = vqe_runner.vqe_run(ansatz=ansatz_elements + [element_to_add, compl_element_to_add],
                                        init_guess_parameters=var_parameters + [0, 0])

        current_energy = result.fun
        delta_e = previous_energy - current_energy

        # get initial guess for the var. params. for the next iteration
        var_parameters = list(result.x)

        if delta_e > 0:
            ansatz_elements.append(element_to_add)
            elements_to_add = [element_to_add]
            if compl_element_to_add is not None:
                ansatz_elements.append(compl_element_to_add)
                elements_to_add.append(compl_element_to_add)

            for elementt in elements_to_add:
                # write iteration data
                try:
                    if elementt.order == 1:
                        element_qubits = [elementt.qubit_1, elementt.qubit_2]
                    elif elementt.order == 2:
                        element_qubits = [elementt.qubit_pair_1, elementt.qubit_pair_2]
                    else:
                        element_qubits = []
                except AttributeError:
                    # this case corresponds to Pauli word excitation
                    element_qubits = elementt.excitations_generators

                gate_count = QasmUtils.gate_count_from_ansatz(ansatz_elements, molecule.n_orbitals)
                df_data.loc[df_count] = {'n': iter_count, 'E': current_energy, 'dE': delta_e,
                                         'error': current_energy - fci_energy,
                                         'n_iters': result['n_iters'], 'cnot_count': gate_count['cnot_count'],
                                         'u1_count': gate_count['u1_count'], 'cnot_depth': gate_count['cnot_depth'],
                                         'u1_depth': gate_count['u1_depth'], 'element': elementt.element,
                                         'element_qubits': element_qubits, 'var_parameters': 0}
                df_count += 1
                df_data['var_parameters'] = list(result.x)[:df_count]
                # df_data['var_parameters'] = var_parameters
                # save data
                save_data(df_data, molecule, time_stamp, ansatz_element_type=ansatz_element_type, frozen_els=frozen_els)

                message = 'Add new element to final ansatz {}. Energy {}. Energy change {}, var. parameters: {}' \
                    .format(element_to_add.element, current_energy, delta_e, var_parameters)
                logging.info(message)
                print(message)
        else:
            message = 'No contribution to energy decrease. Stop adding elements to the final ansatz'
            logging.info(message)
            print(message)
            break

        print('Added element ', ansatz_elements[-1].element)

    # save data. Not required?
    save_data(df_data, molecule, time_stamp, ansatz_element_type=ansatz_element_type)

    # calculate the VQE for the final ansatz
    vqe_runner_final = VQERunner(molecule, backend=QiskitSimBackend, ansatz=ansatz_elements)
    final_result = vqe_runner_final.vqe_run(ansatz=ansatz_elements)
    t = time.time()

    print(final_result)
    print('Ciao')