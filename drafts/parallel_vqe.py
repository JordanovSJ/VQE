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


if __name__ == "__main__":

    molecule = H2
    r = 0.735
    max_n_iterations = 2000
    threshold = 1e-6  # 1e-3 for chemical accuracy

    t0 = time.time()

    ansatz_elements_pool = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()
    # ansatz_elements_pool += FixedAnsatz1(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()

    # vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=[],
    #                               molecule_geometry_params={'distance': r}, optimizer=None)
    # hf_energy = vqe_runner.hf_energy

    # pool = MyPool(3)
    # pool = multiprocessing.pool.Pool(3)
    # new_pool = [pool.apply(add_element_above_threshold, args=(x, molecule, r, threshold, hf_energy)) for x in ansatz_elements_pool]
    # new_pool = pool.starmap_async(add_element_above_threshold, [(x, molecule, r, threshold, hf_energy) for x in ansatz_elements_pool]).get()
    # pool.close()
    # pool.join()

    # @ray.remote
    # def add_element_above_threshold(element):
    #     print('Run VQE')
    #     print(element.element)
    #     local_vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=[element],
    #                                  molecule_geometry_params={'distance': r})
    #     result = local_vqe_runner.vqe_run(max_n_iterations)
    #
    #     if hf_energy - result >= threshold:
    #         print('Add element ', element.element)
    #         return element
    #     else:
    #         return 0
    #
    # ray.init(num_cpus=2)
    #
    # result_ids = []
    # for ansatz_element in ansatz_elements_pool:
    #     print('1')
    #     result_ids.append(add_element_above_threshold.remote(ansatz_element))

    ray.init(num_cpus=2)

    vqe_runners = [VQERunner.remote(molecule, backend=QiskitSimulation, ansatz_elements=[element],
                                    molecule_geometry_params={'distance': r}) for element in ansatz_elements_pool]
    result = ray.get([vqe_runner.vqe_run.remote(max_n_iterations) for vqe_runner in vqe_runners])

    # new_pool = ray.get(result_ids)

    # for i in range(new_pool.count(0)):
    #     new_pool.remove(0)

    print(len(vqe_runners))
