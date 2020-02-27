import sys
sys.path.append('../')

from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF, BeH2
from src.ansatz_elements import UCCGSD, UCCSD, ESD, EGSD, DoubleExchangeAnsatzElement, ExchangeAnsatzElement
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

    accuracy = 1e-5  # 1e-3 for chemical accuracy
    threshold = 1e-7
    max_ansatz_elements = 13

    multithread = True
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>

    LogUtils.log_cofig()

    new_ansatz_element_pool = [DoubleExchangeAnsatzElement([0, 1], [6, 7]), DoubleExchangeAnsatzElement([0, 1], [8, 9]),
                               DoubleExchangeAnsatzElement([0, 1], [10, 13]), DoubleExchangeAnsatzElement([0, 1], [12, 13]),
                               DoubleExchangeAnsatzElement([1, 2], [6, 7]), DoubleExchangeAnsatzElement([0, 1], [8, 9]),
                               DoubleExchangeAnsatzElement([2, 3], [6, 7]), DoubleExchangeAnsatzElement([2, 3], [8, 9]),
                               DoubleExchangeAnsatzElement([2, 3], [10, 11]), DoubleExchangeAnsatzElement([2, 3], [12, 13]),
                               DoubleExchangeAnsatzElement([3, 4], [11, 12]), DoubleExchangeAnsatzElement([4, 5], [6, 7]),
                               DoubleExchangeAnsatzElement([4, 5], [8, 9]), DoubleExchangeAnsatzElement([4, 5], [10, 11]),
                               DoubleExchangeAnsatzElement([4, 5], [12, 13]), DoubleExchangeAnsatzElement([2, 5], [10, 13])
                               ]

    occ_orbitals = [0, 1, 2, 3, 4, 5]
    unocc_orbitals = [8, 9, 10, 11, 12, 13]
    for i in occ_orbitals:
        for j in unocc_orbitals:
            new_ansatz_element_pool.append(ExchangeAnsatzElement(i, j))

    message = 'Length of new pool', len(new_ansatz_element_pool)
    logging.info(message)
    print(message)

    optimizer_options = {'maxcor': 15, 'ftol': 1e-9, 'gtol': 1e-7, 'eps': 1e-02, 'maxfun': 1000, 'maxiter': 1000,
                         'iprint': -1, 'maxls': 20}

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params={'distance': r},
                           optimizer_options=optimizer_options, optimizer='L-BFGS-B')
    hf_energy = vqe_runner.hf_energy

    ansatz_elements = [DoubleExchangeAnsatzElement([2, 3], [10, 11]), DoubleExchangeAnsatzElement([4, 5], [10, 11]),
                       DoubleExchangeAnsatzElement([3, 4], [11, 12]), DoubleExchangeAnsatzElement([2, 5], [10, 13]),
                       DoubleExchangeAnsatzElement([4, 5], [10, 11]), DoubleExchangeAnsatzElement([4, 5], [12, 13]),
                       DoubleExchangeAnsatzElement([2, 3], [6, 7]), DoubleExchangeAnsatzElement([2, 3], [12, 13]),
                       DoubleExchangeAnsatzElement([2, 3], [8, 9]), DoubleExchangeAnsatzElement([2, 3], [12, 13])]
    count = 0
    current_energy = hf_energy
    previous_energy = 0
    var_parameters = [0.1506849672708805, 0.16381713543831095, 0.16777943904771997, -0.16780802212279367,
                       0.13676618057869488, -0.13124487100967253, 0.16771467140452906, -0.08964427599759087,
                      0.16784030725429916, 0.07475576336618228]

    while previous_energy - current_energy >= accuracy or count > max_ansatz_elements:
        count += 1

        print('New cycle ', count)

        previous_energy = current_energy

        # custom optimizer options for this step
        # vqe_runner.optimizer = 'Nelder-Mead'
        # vqe_runner.optimizer_options = {'xatol': 0.0001, 'fatol': 0.0001}

        vqe_runner.optimizer_options = optimizer_options

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