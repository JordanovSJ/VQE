import sys
sys.path.append('../')

from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_elements import UCCGSD, UCCSD, ExchangeAnsatz2
from src.backends import QiskitSimulation
from src.utils import LogUtils, AdaptAnsatzUtils

import logging
import time
import numpy
import pandas
import ray


if __name__ == "__main__":
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>
    molecule = H2
    r = 0.735
    max_n_iterations = 2000

    accuracy = 1e-4
    threshold = 1e-6  # 1e-3 for chemical accuracy
    max_ansatz_elements = 10

    multithread = True
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>

    LogUtils.log_cofig()

    # create a pool of ansatz elements
    initial_ansatz_elements_pool = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()
    # ansatz_elements_pool += FixedAnsatz1(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params={'distance': r})
    hf_energy = vqe_runner.hf_energy

    # get a new ansatz element pool
    elements_energies = AdaptAnsatzUtils.get_ansatz_elements_above_threshold(vqe_runner,
                                                                             initial_ansatz_elements_pool,
                                                                             hf_energy - threshold,
                                                                             multithread=multithread)
    new_ansatz_element_pool = []
    for element, energy in elements_energies:
        new_ansatz_element_pool.append(element)
        message = 'New ansatz element added to updated pool, {}. Delta E = {}'\
            .format(element.fermi_operator, energy - hf_energy)
        logging.info(message)
        print(message)

    exchange_ansatz_element = ExchangeAnsatz2(molecule.n_orbitals, molecule.n_electrons)
    new_ansatz_element_pool.append(exchange_ansatz_element)

    message = 'Length of new pool', len(new_ansatz_element_pool)
    logging.info(message)
    print(message)

    ansatz_elements = []
    count = 0
    current_energy = hf_energy
    previous_energy = 0

    while previous_energy - current_energy >= accuracy or count > max_ansatz_elements:
        count += 1

        print('New cycle ', count)

        previous_energy = current_energy

        element_to_add, current_energy = AdaptAnsatzUtils\
            .get_most_significant_ansatz_element(vqe_runner,
                                                 new_ansatz_element_pool,
                                                 initial_ansatz=ansatz_elements,
                                                 multithread=multithread)

        delta_e = previous_energy - current_energy

        if delta_e > 0:
            ansatz_elements.append(element_to_add)
            message = 'Add new element to final ansatz {}. Energy {}. Energy change {}'\
                .format(element_to_add.fermi_operator, current_energy, delta_e)
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
    current_energy = vqe_runner_final.vqe_run(ansatz_elements=ansatz_elements)
    t = time.time()

    print(current_energy)
    print('Ciao')
