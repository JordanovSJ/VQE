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
from src.q_systems import *
from src.ansatz_element_lists import *
from src.backends import QiskitSim
from src.utils import LogUtils
from src.iter_vqe_utils import *


if __name__ == "__main__":
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<,simulation parameters>>>>>>>>>>>>>>>>>>>>
    r = 0.735
    # theta = 0.538*numpy.pi # for H20
    frozen_els = {'occupied': [], 'unoccupied': []}
    molecule = H4() #(frozen_els=frozen_els)

    ansatz_element_type = 'efficient_fermi_excitation'
    spin_complement = True

    accuracy = 1e-12  # 1e-3 for chemical accuracy
    # threshold = 1e-14
    max_ansatz_elements = 250

    multithread = False
    use_grad = True  # for optimizer
    precompute_commutators = True
    size_patch_commutators = 500
    do_precompute_statevector = True  # for gradients

    init_db = None  # pandas.read_csv("../../results/adapt_vqe_results/LiH_g_adapt_spin_gsdefe_26-Aug-2020.csv")

    if init_db is None:
        ansatz_elements = []
        var_parameters = []
    else:
        ansatz_elements, var_parameters = IterVQEDataUtils.get_ansatz_from_data_frame(init_db, molecule)

        print(len(ansatz_elements))
        print(len(var_parameters))
        assert len(ansatz_elements) == len(var_parameters)

    # var_parameters = list(vqe_runner.vqe_run(ansatz_elements, var_parameters).x)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    LogUtils.log_config()
    logging.info('{}, r={} ,{}'.format(molecule.name, r, ansatz_element_type))

    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

    # create a vqe runner object
    optimizer = 'BFGS'
    optimizer_options = {'gtol': 1e-08}
    vqe_runner = VQERunner(molecule, backend_type=QiskitSim, optimizer=optimizer, optimizer_options=optimizer_options,
                           use_ansatz_gradient=use_grad)
    hf_energy = molecule.hf_energy
    fci_energy = molecule.fci_energy

    # dataFrame to collect the simulation data
    df_data = pandas.DataFrame(columns=['n', 'E', 'dE', 'error', 'n_iters', 'cnot_count', 'u1_count', 'cnot_depth',
                                        'u1_depth', 'element', 'element_qubits', 'var_parameters'])

    # get the pool of ansatz elements
    if spin_complement:
        ansatz_element_pool = SpinComplementGSDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                                           element_type=ansatz_element_type).get_ansatz_elements()
    else:
        ansatz_element_pool = GSDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                             ansatz_element_type=ansatz_element_type).get_ansatz_elements()

    print('Pool len: ', len(ansatz_element_pool))

    if precompute_commutators:
        dynamic_commutators = {}
        print('Calculating commutators')
        for i in range(int(len(ansatz_element_pool)/size_patch_commutators)):
            patch_ansatz_elements = ansatz_element_pool[i*size_patch_commutators:(i+1)*size_patch_commutators]
            patch_dynamic_commutators = IterVQEGradientUtils.calculate_commutators(H_qubit_operator=molecule.jw_qubit_ham,
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

        patch_ansatz_elements = ansatz_element_pool[(int(len(ansatz_element_pool)/size_patch_commutators)) * size_patch_commutators:]
        patch_dynamic_commutators = IterVQEGradientUtils.calculate_commutators(H_qubit_operator=molecule.jw_qubit_ham,
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

    iter_count = 0
    current_energy = hf_energy
    previous_energy = 0
    init_ansatz_length = len(ansatz_elements) #  blalbal

    while previous_energy - current_energy >= accuracy and iter_count <= max_ansatz_elements:
        iter_count += 1

        print('New cycle ', iter_count)

        previous_energy = current_energy

        element_to_add, grad = IterVQEGradientUtils.\
            get_largest_gradient_ansatz_elements(ansatz_element_pool, molecule, backend_type=vqe_runner.backend_type,
                                                 var_parameters=var_parameters, ansatz=ansatz_elements,
                                                 multithread=multithread, dynamic_commutators=dynamic_commutators)[0]
        print(element_to_add.element)

        result = vqe_runner.vqe_run(ansatz=ansatz_elements + [element_to_add], initial_var_parameters=var_parameters + [0])

        current_energy = result.fun
        delta_e = previous_energy - current_energy

        # get initial guess for the var. params. for the next iteration
        var_parameters = list(result.x)

        if delta_e > 0:

            ansatz_elements.append(element_to_add)

            # write iteration data
            try:
                if element_to_add.order == 1:
                    element_qubits = [element_to_add.qubit_1, element_to_add.qubit_2]
                elif element_to_add.order == 2:
                    element_qubits = [element_to_add.qubit_pair_1, element_to_add.qubit_pair_2]
                else:
                    element_qubits = []
            except AttributeError:
                # this case corresponds to Pauli word excitation
                element_qubits = element_to_add.excitation_generator

            gate_count = QasmUtils.gate_count_from_ansatz_elements(ansatz_elements, molecule.n_orbitals)
            df_data.loc[iter_count] = {'n': iter_count, 'E': current_energy, 'dE': delta_e, 'error': current_energy-fci_energy,
                                       'n_iters': result['n_iters'], 'cnot_count': gate_count['cnot_count'],
                                       'u1_count': gate_count['u1_count'], 'cnot_depth': gate_count['cnot_depth'],
                                       'u1_depth': gate_count['u1_depth'], 'element': element_to_add.element,
                                       'element_qubits': element_qubits, 'var_parameters': 0}
            df_data['var_parameters'] = list(result.x)[init_ansatz_length:]
            # df_data['var_parameters'] = var_parameters
            # save data
            save_data(df_data, molecule, time_stamp, ansatz_element_type=ansatz_element_type, frozen_els=frozen_els)

            message = 'Add new element to final ansatz {}. Energy {}. Energy change {}, Grad{}, var. parameters: {}' \
                .format(element_to_add.element, current_energy, delta_e, grad, var_parameters)
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
    vqe_runner_final = VQERunner(molecule, backend_type=QiskitSim, ansatz=ansatz_elements)
    final_result = vqe_runner_final.vqe_run(ansatz=ansatz_elements)
    t = time.time()

    print(final_result)
    print('Ciao')
