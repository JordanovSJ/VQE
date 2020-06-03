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
from src.molecules import *
from src.ansatz_element_lists import *
from src.backends import QiskitSimulation
from src.utils import LogUtils, AdaptAnsatzUtils


if __name__ == "__main__":
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>
    molecule = H2
    r = 0.735
    # theta = 0.538*numpy.pi # for H20
    molecule_params = {'distance': r}  #, 'theta': theta}

    ansatz_element_type = 'efficient_fermi_excitation'
    # ansatz_element_type = 'qubit_excitation'

    accuracy = 1e-6  # 1e-3 for chemical accuracy
    threshold = 1e-7
    max_ansatz_elements = 30

    multithread = True
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>

    LogUtils.log_cofig()
    logging.info('{}, r={} ,{}'.format(molecule.name, r, ansatz_element_type))

    # create a pool of ansatz elements
    initial_ansatz_elements_pool = SDElements(molecule.n_orbitals, molecule.n_electrons,
                                              element_type=ansatz_element_type).get_double_excitations()

    # create a vqe runner object
    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params=molecule_params)
    hf_energy = vqe_runner.hf_energy

    # New pool
    # get a new ansatz element pool, from elements that decrease <H> by at least dE = threshold
    new_ansatz_element_pool = []
    elements_results = AdaptAnsatzUtils.get_ansatz_elements_above_threshold(vqe_runner,
                                                                            initial_ansatz_elements_pool,
                                                                            hf_energy - threshold,
                                                                            multithread=multithread)
    for element, result in elements_results:
        new_ansatz_element_pool.append(element)
        message = 'New ansatz element added to updated pool, {}. Delta E = {}'\
            .format(element.element, result.fun - hf_energy)
        logging.info(message)
        print(message)

    new_ansatz_element_pool += SDElements(molecule.n_orbitals, molecule.n_electrons,
                                          element_type=ansatz_element_type).get_single_excitations()

    message = 'Length of new pool', len(new_ansatz_element_pool)
    logging.info(message)
    print(message)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ansatz_elements = []
    var_parameters = []
    count = 0
    current_energy = hf_energy
    fci_energy = vqe_runner.fci_energy
    previous_energy = 0

    # dataFrame to collect the simulation data
    df_data = pandas.DataFrame(columns=['n', 'E', 'dE', 'error', 'n_iters', 'cnot_count', 'u1_count', 'cnot_depth', 'u1_depth'])

    while previous_energy - current_energy >= accuracy and count <= max_ansatz_elements:
        count += 1

        print('New cycle ', count)

        previous_energy = current_energy

        element_to_add, result = AdaptAnsatzUtils.get_most_significant_ansatz_element(vqe_runner,
                                                                                      new_ansatz_element_pool,
                                                                                      initial_var_parameters=var_parameters,
                                                                                      initial_ansatz=ansatz_elements,
                                                                                      multithread=multithread)
        current_energy = result.fun
        # get initial guess for the var. params. for the next iteration
        var_parameters = list(result.x) + list(numpy.zeros(element_to_add.n_var_parameters))
        delta_e = previous_energy - current_energy

        if delta_e > 0:
            ansatz_elements.append(element_to_add)

            # add step data
            gate_count = QasmUtils.gate_count_from_ansatz_elements(ansatz_elements, molecule.n_orbitals)
            df_data.loc[count] = {'n': count, 'E': current_energy, 'dE': delta_e, 'error': current_energy-fci_energy,
                                  'n_iters': result['n_iters'], 'cnot_count': gate_count['cnot_count'],
                                  'u1_count': gate_count['u1_count'], 'cnot_depth': gate_count['cnot_depth'],
                                  'u1_depth': gate_count['u1_depth']}

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

    # save the data to a secv file
    time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    try:
        df_data.to_csv('../../results/adapt_vqe_results/{}_{}.csv'.format(molecule.name, time_stamp))
    except FileNotFoundError:
        try:
            df_data.to_csv('results/adapt_vqe_results/{}_{}.csv'.format(molecule.name, time_stamp))
        except FileNotFoundError as fnf:
            print(fnf)

    # calculate the VQE for the final ansatz
    vqe_runner_final = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params={'distance': r}
                                 , ansatz_elements=ansatz_elements)
    final_result = vqe_runner_final.vqe_run(ansatz_elements=ansatz_elements)
    t = time.time()

    print(final_result)
    print('Ciao')
