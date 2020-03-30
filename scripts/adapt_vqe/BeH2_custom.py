import sys
sys.path.append('../')

from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF, BeH2
from src.ansatz_elements import UCCGSD, UCCSD, ESD, EGSD, DoubleExchange, SingleExchange, DoubleExcitation, SingleExcitation
from src.backends import QiskitSimulation
from src.utils import LogUtils, AdaptAnsatzUtils

import logging
import time
import numpy
import pandas
import ray


if __name__ == "__main__":
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>
    molecule = BeH2
    r = 1.3264
    max_n_iterations = 2000

    accuracy = 1e-6  # 1e-3 for chemical accuracy
    threshold = 1e-7
    max_ansatz_elements = 13

    multithread = True
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>

    LogUtils.log_cofig()

    # create a pool of ansatz elements
    initial_ansatz_elements_pool = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_double_excitation_list()

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params={'distance': r},)
    hf_energy = vqe_runner.hf_energy

    # get a new ansatz element pool
    elements_results = AdaptAnsatzUtils.get_ansatz_elements_above_threshold(vqe_runner,
                                                                            initial_ansatz_elements_pool,
                                                                            hf_energy - threshold,
                                                                            multithread=multithread)
    new_ansatz_element_pool = []
    for element, result in elements_results:
        new_ansatz_element_pool.append(element)
        message = 'New ansatz element added to updated pool, {}. Delta E = {}' \
            .format(element.element, result.fun - hf_energy)
        logging.info(message)
        print(message)

    new_ansatz_element_pool += UCCSD(molecule.n_orbitals, molecule.n_electrons).get_single_excitation_list()

    message = 'Length of new pool', len(new_ansatz_element_pool)
    logging.info(message)
    print(message)

    ansatz_elements = [DoubleExcitation([4, 5], [10, 11]), DoubleExcitation([3, 4], [11, 12]),
                       DoubleExcitation([2, 5], [10, 13]), DoubleExcitation([2, 3], [10, 11]),
                       DoubleExcitation([4, 5], [12, 13]), DoubleExcitation([2, 3], [6, 7]),
                       DoubleExcitation([2, 5], [11, 12]), DoubleExcitation([3, 4], [10, 13]),
                       DoubleExcitation([2, 3], [8, 9]), DoubleExcitation([2, 3], [12, 13]),
                       SingleExcitation(3, 11)]
    count = 0
    current_energy = hf_energy
    previous_energy = 0
    var_parameters = [0.07524622301174176, 0.048566337341091864, 0.04878703191135014, 0.05903878703114574,
                      0.04106504718231557, 0.05408221080970733, -0.042467237248041294, -0.04153775187828831,
                      0.05425409011047975, 0.028003214545926677, -0.011157935689415909, 0.0]

    while previous_energy - current_energy >= accuracy or count > max_ansatz_elements:
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
            message = 'Add new element to final ansatz {}. Energy {}. Energy change {}, var. parameters: {}'\
                .format(element_to_add.element, current_energy, delta_e, var_parameters)
            logging.info(message)
            print(message)
        else:
            message = 'No contribution to energy decrease. Stop adding elements to the final ansatz'
            logging.info(message)
            print(message)
            break

        print('Added element ', ansatz_elements[-1].element)

    # calculate the VQE for the final ansatz
    vqe_runner_final = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params={'distance': r}
                                 , ansatz_elements=ansatz_elements)
    final_result = vqe_runner_final.vqe_run(ansatz_elements=ansatz_elements)
    t = time.time()

    print(final_result)
    print('Ciao')
