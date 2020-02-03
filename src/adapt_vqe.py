from src.vqe import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_elements import UCCGSD, UCCSD, FixedAnsatz1
from src.backends import QiskitSimulation
import logging
import time
import numpy
import pandas
import multiprocessing


if __name__ == "__main__":

    molecule = HF
    r = 0.995
    max_n_iterations = 2000
    threshold = 1e-6  # 1e-3 for chemical accuracy

    t0 = time.time()

    ansatz_elements_pool = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()
    # ansatz_elements_pool += FixedAnsatz1(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=[],
                           molecule_geometry_params={'distance': r}, optimizer=None)
    hf_energy = vqe_runner.vqe_run(max_n_iterations)

    new_pool = []


    def add_element_above_threshold(local_ansatz_element):
        local_vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=[local_ansatz_element],
                                     molecule_geometry_params={'distance': r})
        local_result = local_vqe_runner.vqe_run(max_n_iterations)

        if hf_energy - local_result >= threshold:
            new_pool.append(ansatz_element)
            print('Add element ', ansatz_element.element)


    # # First pick only the excitations that contribute above a threshold
    # for i, ansatz_element in enumerate(ansatz_elements_pool):
    #     print(i)
    #     vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=[ansatz_element],
    #                            molecule_geometry_params={'distance': r})
    #     result = vqe_runner.vqe_run(max_n_iterations)
    #
    #     if hf_energy-result >= threshold:
    #         new_pool.append(ansatz_elements_pool[i])
    #         print('Add element ', ansatz_element.element)

    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    pool.map_async(add_element_above_threshold, [ansatz_element for ansatz_element in ansatz_elements_pool])

    ansatz_elements_pool = new_pool
    print('Length of new pool', len(ansatz_elements_pool))

    ansatz_elements = []
    count = 0
    energy = hf_energy

    while count < min(5, len(ansatz_elements_pool)):
        count += 1

        previous_energy = energy
        vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                               molecule_geometry_params={'distance': r}, optimizer=None)

        energy = vqe_runner.vqe_run(max_n_iterations)

        print('New cycle ', count, 'New energy ', energy, ' Delta energy ', energy - previous_energy)

        delta_e = 0
        index_element_to_add = 0

        # TODO this can be parallelized
        for i, ansatz_element in enumerate(ansatz_elements_pool):
            print(i)
            if ansatz_element.element_type == 'excitation':
                optimizer = None
            else:
                optimizer = 'Nelder-Mead'

            vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=[ansatz_element],
                                   molecule_geometry_params={'distance': r}, optimizer=optimizer)

            result = vqe_runner.vqe_run(max_n_iterations)

            # Bad criteria
            if result - energy < delta_e:
                delta_e = result - energy
                index_element_to_add = i

        ansatz_elements.append(ansatz_elements_pool[index_element_to_add])
        # TODO this may not be necessary
        del ansatz_elements_pool[index_element_to_add]

        print('Added element ', ansatz_elements[-1].element)

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=ansatz_elements,
                           molecule_geometry_params={'distance': r}, optimizer=None)

    energy = vqe_runner.vqe_run(max_n_iterations)
    t = time.time()

    print(energy)
    print('Run time ', time.time() - t0)
    print('Ciao')
