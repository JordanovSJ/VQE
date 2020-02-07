from src.vqe import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_elements import UCCGSD, UCCSD, FixedAnsatz1
from src.backends import QiskitSimulation
from src.utils import LogUtils

import logging
import time
import numpy
import pandas
import multiprocessing
import ray


@ray.remote
def mark_element_above_threshold(ansatz_element, vqe_runner, threshold):
    print('Run VQE')
    t0 = time.time()

    energy = vqe_runner.vqe_run_parallel([ansatz_element])

    delta_e = energy - vqe_runner.hf_energy
    print('VQE time: ', time.time() - t0, 'Energy change ', delta_e)

    if (-delta_e) >= threshold:
        print('Add element ', ansatz_element.element)
        return ansatz_element
    else:
        return 0


if __name__ == "__main__":

    molecule = HF
    r = 0.995
    max_n_iterations = 2000
    threshold = 1e-6  # 1e-3 for chemical accuracy
    n_elements = 10

    LogUtils.log_cofig()

    ansatz_elements_pool = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()
    # ansatz_elements_pool += FixedAnsatz1(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params={'distance': r})
    hf_energy = vqe_runner.hf_energy

    ray.init(num_cpus=4)
    result_ids = [mark_element_above_threshold.remote(ansatz_element, vqe_runner=vqe_runner, threshold=threshold) for
                  ansatz_element in ansatz_elements_pool]

    new_pool = ray.get(result_ids)

    for i in range(new_pool.count(0)):
        new_pool.remove(0)

    ansatz_elements_pool = new_pool
    print('Length of new pool', len(ansatz_elements_pool))

    ansatz_elements = []
    count = 0
    energy = hf_energy

    while count < min(5, len(ansatz_elements_pool)):
        count += 1

        previous_energy = energy
        energy = vqe_runner.vqe_run(ansatz_elements=ansatz_elements)

        print('New cycle ', count, 'New energy ', energy, ' Delta energy ', energy - previous_energy)

        delta_e = 0
        index_element_to_add = 0

        # TODO this can be parallelized
        for i, ansatz_element in enumerate(ansatz_elements_pool):
            print(i)

            result = vqe_runner.vqe_run(ansatz_elements=ansatz_elements+[ansatz_element])

            # Bad criteria
            if result - energy < delta_e:
                delta_e = result - energy
                index_element_to_add = i

        ansatz_elements.append(ansatz_elements_pool[index_element_to_add])

        print('Added element ', ansatz_elements[-1].element)

    vqe_runner_final = VQERunner(molecule, backend=QiskitSimulation, molecule_geometry_params={'distance': r}
                                 , ansatz_elements=ansatz_elements)
    energy = vqe_runner_final.vqe_run(ansatz_elements=ansatz_elements)
    t = time.time()

    print(energy)
    print('Ciao')
