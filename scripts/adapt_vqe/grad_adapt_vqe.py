import logging
import time
import numpy
import pandas
import ray
import pandas
import datetime

import sys
sys.path.append('../')

from src.vqe_runner import VQERunner
from src.q_systems import *
from src.ansatz_element_lists import *
from src.backends import QiskitSimulation
from src.utils import LogUtils
from src.adapt_utils import GradAdaptUtils


def save_data(df_data, molecule, time_stamp, ansatz_element_type=None):
    if ansatz_element_type is None:
        ansatz_element_type = 'unspecified'

    try:
        df_data.to_csv('../../results/adapt_vqe_results/{}_{}_{}.csv'.format(molecule.name, ansatz_element_type, time_stamp))
    except FileNotFoundError:
        try:
            df_data.to_csv('results/adapt_vqe_results/{}_{}_{}.csv'.format(molecule.name, ansatz_element_type, time_stamp))
        except FileNotFoundError as fnf:
            print(fnf)


if __name__ == "__main__":
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<,simulation parameters>>>>>>>>>>>>>>>>>>>>
    molecule = HF
    r = 0.995
    # theta = 0.538*numpy.pi # for H20
    molecule_params = {'distance': r}  #, 'theta': theta}

    ansatz_element_type = 'efficient_fermi_excitation'
    #ansatz_element_type = 'qubit_excitation'

    accuracy = 1e-8  # 1e-3 for chemical accuracy
    # threshold = 1e-14
    max_ansatz_elements = 45

    multithread = True
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    LogUtils.log_cofig()
    logging.info('{}, r={} ,{}'.format(molecule.name, r, ansatz_element_type))

    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

    # create a vqe runner object
    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params=molecule_params)
    hf_energy = vqe_runner.hf_energy
    fci_energy = vqe_runner.fci_energy

    # dataFrame to collect the simulation data
    df_data = pandas.DataFrame(columns=['n', 'E', 'dE', 'error', 'n_iters', 'cnot_count', 'u1_count', 'cnot_depth',
                                        'u1_depth', 'element', 'element_qubits', 'var_parameters'])
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>?>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # get single excitaitons
    ansatz_element_pool = SDElements(molecule.n_orbitals, molecule.n_electrons,
                                     element_type=ansatz_element_type).get_ansatz_elements()

    message = 'Length of new pool', len(ansatz_element_pool)
    logging.info(message)
    print(message)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ansatz_elements = []
    var_parameters = []
    count = 0
    current_energy = hf_energy
    previous_energy = 0

    while previous_energy - current_energy >= accuracy and count <= max_ansatz_elements:
        count += 1

        print('New cycle ', count)

        previous_energy = current_energy

        element_to_add, grad = GradAdaptUtils.\
            get_most_significant_ansatz_element(ansatz_element_pool, vqe_runner.jw_qubit_ham, vqe_runner.n_qubits,
                                                vqe_runner.n_electrons, vqe_runner.backend,
                                                initial_var_parameters=var_parameters, initial_ansatz=ansatz_elements,
                                                multithread=multithread)

        result = vqe_runner.vqe_run(ansatz_elements=ansatz_elements+[element_to_add],
                                    initial_var_parameters=var_parameters + list(numpy.zeros(element_to_add.n_var_parameters)))

        current_energy = result.fun
        delta_e = previous_energy - current_energy

        # get initial guess for the var. params. for the next iteration
        var_parameters = list(result.x)

        if delta_e > 0:
            ansatz_elements.append(element_to_add)

            # write iteration data
            if element_to_add.order == 1:
                element_qubits = [element_to_add.qubit_1, element_to_add.qubit_2]
            elif element_to_add.order == 2:
                element_qubits = [element_to_add.qubit_pair_1, element_to_add.qubit_pair_2]
            else:
                element_qubits = []

            gate_count = QasmUtils.gate_count_from_ansatz_elements(ansatz_elements, molecule.n_orbitals)
            df_data.loc[count] = {'n': count, 'E': current_energy, 'dE': delta_e, 'error': current_energy-fci_energy,
                                  'n_iters': result['n_iters'], 'cnot_count': gate_count['cnot_count'],
                                  'u1_count': gate_count['u1_count'], 'cnot_depth': gate_count['cnot_depth'],
                                  'u1_depth': gate_count['u1_depth'], 'element': element_to_add.element,
                                  'element_qubits': element_qubits, 'var_parameters': 0}
            df_data['var_parameters'] = result.x
            # df_data['var_parameters'] = var_parameters
            # save data
            save_data(df_data, molecule, time_stamp, ansatz_element_type=ansatz_element_type)

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
    vqe_runner_final = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params={'distance': r}
                                 , ansatz_elements=ansatz_elements)
    final_result = vqe_runner_final.vqe_run(ansatz_elements=ansatz_elements)
    t = time.time()

    print(final_result)
    print('Ciao')
