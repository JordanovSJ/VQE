from src.vqe import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_elements import UCCGSD, UCCSD, FixedAnsatz1
from src.backends import QiskitSimulation
from src.utils import LogUtils

import logging
import time
import numpy
import pandas
import ray


if __name__ == "__main__":
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>
    molecule = HF
    r = 0.995
    max_n_iterations = 2000

    accuracy = 1e-3
    threshold = 1e-6  # 1e-3 for chemical accuracy
    max_ansatz_elements = 10

    multithread = True
    n_cpus = 4
    # <<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>

    LogUtils.log_cofig()

    # create a pool of ansatz elements
    initial_ansatz_elements_pool = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()
    # ansatz_elements_pool += FixedAnsatz1(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params={'distance': r})
    hf_energy = vqe_runner.hf_energy

    # calculate the energy contribution of each ansatz element
    if multithread:
        ray.init(num_cpus=n_cpus)
        ids_energies = [
            [i, ray.get(vqe_runner.vqe_run_multithread.remote(self=vqe_runner, ansatz_elements=[element]))]
            for i, element in enumerate(initial_ansatz_elements_pool)]
    else:
        ids_energies = [
            [i, vqe_runner.vqe_run(ansatz_elements=[element])]
            for i, element in enumerate(initial_ansatz_elements_pool)]

    # create a new pool of ansatz elements
    new_ansatz_element_pool = []
    for i, current_energy in ids_energies:
        delta_e = hf_energy - current_energy
        if delta_e >= threshold:
            element = initial_ansatz_elements_pool[i]
            new_ansatz_element_pool.append(element)
            message = 'New ansatz element added to updated pool, {}. Delta E = {}'.format(element.fermi_operator, delta_e)
            logging.info(message)
            print(message)

    message = 'Length of new pool', len(new_ansatz_element_pool)
    logging.info(message)
    print(message)

    ansatz_elements = []
    count = 0
    current_energy = hf_energy
    previous_energy = 0

    # TODO suitable condition?
    while previous_energy - current_energy >= accuracy or count > max_ansatz_elements:  # count < min(5, len(initial_ansatz_elements_pool)):
        count += 1

        # energy = vqe_runner.vqe_run(ansatz_elements=ansatz_elements)

        print('New cycle ', count)

        # calculate the energy contribution of each ansatz element
        if multithread:
            ids_energies = [
                [i, ray.get(vqe_runner.vqe_run_multithread.remote(self=vqe_runner, ansatz_elements=ansatz_elements + [element]))]
                for i, element in enumerate(new_ansatz_element_pool)]
        else:
            ids_energies = [
                [i, vqe_runner.vqe_run(ansatz_elements=ansatz_elements + [element])]
                for i, element in enumerate(new_ansatz_element_pool)]

        previous_energy = current_energy
        delta_e = 0
        # find the element with greatest contribution
        for i, energy in ids_energies:
            if previous_energy - energy > delta_e:
                delta_e = previous_energy - energy
                current_energy = energy
                element_to_add_index = i

        if delta_e > 0:
            element = new_ansatz_element_pool[element_to_add_index]
            ansatz_elements.append(element)
            message = 'Add new element to final ansatz {}. Energy {}. Energy change {}'\
                .format(element.fermi_operator, current_energy, delta_e)
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
