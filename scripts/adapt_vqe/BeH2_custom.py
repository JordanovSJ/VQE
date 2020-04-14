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
    r = 1.316
    max_n_iterations = 2000

    accuracy = 1e-6  # 1e-3 for chemical accuracy
    threshold = 1e-7
    max_ansatz_elements = 13

    multithread = True
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>

    LogUtils.log_cofig()

    logging.info('Continued simulation: adapt_ESD BeH2 rescaled, corrected, parity')

    pool_d_exc_qubits = [[[0, 1], [6, 7]], [[0, 1], [8, 9]], [[0, 1], [10, 11]], [[0, 1], [12, 13]], [[0, 3], [6, 7]],
                          [[0, 3], [8, 9]], [[0, 3], [10, 11]], [ [0, 3], [12, 13]], [[0, 4], [10, 12]], [[0, 5], [10, 13]],
                          [[0, 5], [11, 12]], [[1, 2], [6, 7]], [[1, 2], [8, 9]], [[1, 2], [10, 11]], [[1, 2], [12, 13]],
                          [ [1, 4], [10, 13]], [ [1, 4], [11, 12]], [[1, 5], [11, 13]], [[2, 3], [6, 7]], [ [2, 3], [8, 9]],
                          [[2, 3], [10, 11]], [[2, 3], [12, 13]], [[2, 4], [10, 12]], [[2, 5], [10, 13]], [ [2, 5], [11, 12]],
                          [[3, 4], [10, 13]], [[3, 4], [11, 12]], [[3, 5], [11, 13]], [[4, 5], [6, 7]], [[4, 5], [8, 9]],
                          [[4, 5], [10, 11]], [[4, 5], [12, 13]]]

    new_ansatz_element_pool = \
        [DoubleExchange(qubits[0], qubits[1], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True)
         for qubits in pool_d_exc_qubits]

    new_ansatz_element_pool += UCCSD(molecule.n_orbitals, molecule.n_electrons).get_single_excitations()

    message = 'Length of new pool', len(new_ansatz_element_pool)
    logging.info(message)
    print(message)

    ansatz_elements = [DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
                       DoubleExchange([2, 3], [10, 11], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
                       DoubleExchange([3, 4], [11, 12], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
                       DoubleExchange([4, 5], [12, 13], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
                       DoubleExchange([2, 3], [6, 7], rescaled_parameter=True, d_exc_correction=True, parity_dependence=True),
                       DoubleExchange([3, 4], [10, 13] , rescaled_parameter=True, d_exc_correction=True,
                                      parity_dependence=True),

                       DoubleExchange([2, 5], [11, 12], rescaled_parameter=True, d_exc_correction=True,
                                      parity_dependence=True),
                       DoubleExchange([2, 5], [10, 13], rescaled_parameter=True, d_exc_correction=True,
                                      parity_dependence=True),
                       DoubleExchange([2, 3], [8, 9], rescaled_parameter=True, d_exc_correction=True,
                                      parity_dependence=True),
                       DoubleExchange([2, 3], [12, 13], rescaled_parameter=True, d_exc_correction=True,
                                      parity_dependence=True),
                       DoubleExchange([3, 5], [11, 13], rescaled_parameter=True, d_exc_correction=True,
                                      parity_dependence=True),
                       DoubleExchange([2, 4], [10, 12], rescaled_parameter=True, d_exc_correction=True,
                                      parity_dependence=True),
                       DoubleExchange([4, 5], [10, 11], rescaled_parameter=True, d_exc_correction=True,
                                      parity_dependence=True),
                       DoubleExchange([3, 4], [11, 12], rescaled_parameter=True, d_exc_correction=True,
                                      parity_dependence=True),
                       SingleExchange( 3, 12)
                       ]

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params={'distance': r}, )
    hf_energy = vqe_runner.hf_energy

    count = 0
    current_energy = hf_energy
    previous_energy = 0
    var_parameters =[0.02542504343149286, 0.022800826074214103, 0.019658085166753464, 0.016359315630631654,
                     0.020856005034729494, -0.015290856033162612, -0.015482320072768922, 0.017327150133654004,
                     0.020917710627941843, 0.0089730955422877, 0.0030479645826446636, 0.003004512035924215,
                     -0.0015, -0.003517727107124633, -0.014030400855921022, 0.0]

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
