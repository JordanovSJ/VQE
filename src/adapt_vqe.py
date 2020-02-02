from src.vqe import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_elements import UCCGSD, UCCSD, FixedAnsatz1
from src.backends import QiskitSimulation
import logging
import time
import numpy
import pandas

if __name__ == "__main__":

    molecule = HF
    r = 0.995
    max_n_iterations = 2000

    ansatz_elements_pool = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()
    # ansatz_elements_pool += FixedAnsatz1(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()

    ansatz_elements = []
    count = 0

    vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=[],
                           molecule_geometry_params={'distance': r}, optimizer=None)
    energy = vqe_runner.vqe_run(max_n_iterations)

    while count < 2:
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

            if ansatz_element.element_type == 'excitation':
                optimizer = None
            else:
                optimizer = 'Nelder-Mead'

            vqe_runner = VQERunner(molecule, backend=QiskitSimulation, ansatz_elements=[ansatz_element],
                                   molecule_geometry_params={'distance': r}, optimizer=optimizer)

            result = vqe_runner.vqe_run(max_n_iterations)
            if result - energy < delta_e:
                delta_e = result - energy
                index_element_to_add = i

        ansatz_elements.append(ansatz_elements_pool[index_element_to_add])
        # TODO this may not be necessary
        del ansatz_elements_pool[index_element_to_add]

        print('Added element ', ansatz_elements[-1].element)

    t0 = time.time()
    t = time.time()

    print(result)

    print('Ciao')
