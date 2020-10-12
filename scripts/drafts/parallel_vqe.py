from src.vqe_runner import VQERunner
from src.q_systems import H2, LiH, HF
from src.ansatz_elements import UCCGSD, UCCSD
from src.backends import QiskitSim
import logging
import time
import numpy
import pandas
import ray


if __name__ == "__main__":

    molecule = HF
    r = 0.995
    max_n_iterations = 2000
    threshold = 1e-6  # 1e-3 for chemical accuracy

    t0 = time.time()

    ansatz_elements_pool = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()

    vqe_runner = VQERunner(molecule, backend=QiskitSim, molecule_geometry_params={'distance': r})

    ray.init(num_cpus=4)

    result_ids = [[i, vqe_runner.vqe_run_multithread.remote(self=vqe_runner, ansatz=[ansatz_element])]
                  for i, ansatz_element in enumerate(ansatz_elements_pool)]

    results = [[result_id[0], ray.get(result_id[1])] for result_id in result_ids]

    print(len(results))
    print(results)
