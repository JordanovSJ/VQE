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
from src.backends import QiskitSim
from src.utils import LogUtils
from src.adapt_utils import EnergyAdaptUtils


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
    r = 1.316
    # theta = 0.538*numpy.pi # for H20
    molecule = H2()

    # ansatz_element_type = 'efficient_fermi_excitation'
    ansatz_element_type = 'qubit_excitation'

    accuracy = 1e-13  # 1e-3 for chemical accuracy
    threshold = 1e-14
    max_ansatz_elements = 40

    multithread = True
    use_grad = False
    compute_exc_mtrx = use_grad
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

    # create a vqe runner object
    vqe_runner = VQERunner(molecule, backend=QiskitSim, use_ansatz_gradient=use_grad)
    hf_energy = molecule.hf_energy
    fci_energy = molecule.fci_energy

    # dataFrame to collect the simulation data
    df_data = pandas.DataFrame(columns=['n', 'E', 'dE', 'error', 'n_iters', 'cnot_count', 'u1_count', 'cnot_depth',
                                        'u1_depth', 'element', 'element_qubits', 'var_parameters'])
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>

    LogUtils.log_config()
    logging.info('{}, r={} ,{}'.format(molecule.name, r, ansatz_element_type))

    # create a pool of ansatz elements
    initial_ansatz_elements_pool = GSDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                                  ansatz_element_type=ansatz_element_type,
                                                  compute_exc_mtrx=compute_exc_mtrx).get_double_excitations()

    # New pool
    # get a new ansatz element pool, from elements that decrease <H> by at least dE = threshold
    new_ansatz_element_pool = []
    elements_results = EnergyAdaptUtils.elements_below_full_vqe_energy_reduction_threshold(vqe_runner,
                                                                                           initial_ansatz_elements_pool,
                                                                                           hf_energy - threshold,
                                                                                           multithread=multithread)
    for element, result in elements_results:
        new_ansatz_element_pool.append(element)
        message = 'New ansatz element added to updated pool, {}. Delta E = {}'\
            .format(element.element, result.fun - hf_energy)
        logging.info(message)
        print(message)

    # get single excitaitons
    new_ansatz_element_pool += SDExcitations(molecule.n_orbitals, molecule.n_electrons,
                                             element_type=ansatz_element_type).get_single_excitations()

    message = 'Length of new pool', len(new_ansatz_element_pool)
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

        element_to_add, result = EnergyAdaptUtils.largest_individual_element_vqe_energy_reduction(vqe_runner,
                                                                                                  new_ansatz_element_pool,
                                                                                                  var_parameters=var_parameters,
                                                                                                  ansatz=ansatz_elements,
                                                                                                  multithread=multithread)
        current_energy = result.fun

        # TODO works only if all elements have single var par
        # get initial guess for the var. params. for the next iteration
        # var_parameters = list(result.x) + list(numpy.zeros(element_to_add.n_var_parameters))
        var_parameters = list(result.x)
        delta_e = previous_energy - current_energy

        if delta_e > 0:
            ansatz_elements.append(element_to_add)

            # step data
            if element_to_add.order == 1:
                element_qubits = [element_to_add.qubit_1, element_to_add.qubit_2]
            elif element_to_add.order == 2:
                element_qubits = [element_to_add.qubit_pair_1, element_to_add.qubit_pair_2]
            else:
                element_qubits = []

            gate_count = QasmUtils.gate_count_from_ansatz(ansatz_elements, molecule.n_orbitals)
            df_data.loc[count] = {'n': count, 'E': current_energy, 'dE': delta_e, 'error': current_energy-fci_energy,
                                  'n_iters': result['n_iters'], 'cnot_count': gate_count['cnot_count'],
                                  'u1_count': gate_count['u1_count'], 'cnot_depth': gate_count['cnot_depth'],
                                  'u1_depth': gate_count['u1_depth'], 'element': element_to_add.element,
                                  'element_qubits': element_qubits, 'var_parameters': 0}
            df_data['var_parameters'] = result.x
            # df_data['var_parameters'] = var_parameters
            # save data
            save_data(df_data, molecule, time_stamp, ansatz_element_type=ansatz_element_type)

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
    vqe_runner_final = VQERunner(molecule, backend=QiskitSim, ansatz=ansatz_elements)
    final_result = vqe_runner_final.vqe_run(ansatz=ansatz_elements)
    t = time.time()

    print(final_result)
    print('Ciao')
