from src.vqe import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_elements import UCCGSD, UCCSD, FixedAnsatz1
from src.backends import QiskitSimulation
import logging
import time
import numpy
import pandas
import multiprocessing
import ray


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


# @ray.remote
def add_element_above_threshold(ansatz_element, molecule, r, threshold, hf_energy):
    print('Run VQE')
    print(ansatz_element.element)
    local_vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=[ansatz_element],
                                 molecule_geometry_params={'distance': r})
    result = local_vqe_runner.vqe_run(max_n_iterations)

    if hf_energy - result >= threshold:
        print('Add element ', ansatz_element.element)
        return ansatz_element
    else:
        return 0


if __name__ == "__main__":

    molecule = H2
    r = 0.735
    max_n_iterations = 2000
    threshold = 1e-6  # 1e-3 for chemical accuracy

    t0 = time.time()

    ansatz_elements_pool = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()
    # ansatz_elements_pool += FixedAnsatz1(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=[],
                           molecule_geometry_params={'distance': r}, optimizer=None)
    hf_energy = vqe_runner.hf_energy

    # pool = MyPool(3)
    # pool = multiprocessing.pool.Pool(3)
    # new_pool = [pool.apply(add_element_above_threshold, args=(x, molecule, r, threshold, hf_energy)) for x in ansatz_elements_pool]
    # new_pool = pool.starmap_async(add_element_above_threshold, [(x, molecule, r, threshold, hf_energy) for x in ansatz_elements_pool]).get()
    # pool.close()
    # pool.join()

    # ray.init(num_cpus = 4)

    result_ids = []
    for ansatz_element in ansatz_elements_pool:
        # result_ids.append(add_element_above_threshold.remote(ansatz_element, molecule, r, threshold, hf_energy))
        result_ids.append(add_element_above_threshold(ansatz_element, molecule, r, threshold, hf_energy))


    # new_pool = ray.get(result_ids)
    new_pool = result_ids

    for i in range(new_pool.count(0)):
        new_pool.remove(0)

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
