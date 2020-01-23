from src.vqe_runner import VQERunner
from src.molecules import H2, LiH, HF
from src.ansatz_types import UCCGSD, UCCSD, FixedAnsatz1
from src.backends import QiskitSimulation
import logging
import time
import numpy
import pandas

if __name__ == "__main__":

    molecule = H2
    max_n_iterations = 2000

    ansatz_elements = FixedAnsatz1(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()
    # ansatz_elements = [ansatz_elements[0][:5], ansatz_elements[1]]

    values = []
    for i in range(50):
        r = 0.4 + i*0.02
        print('index ', i, 'R = ', r)
        vqe_runner = VQERunner(molecule, backend=QiskitSimulation, excitation_list=ansatz_elements, molecule_geometry_params={'distance': r})
        # t0 = time.time()
        result = vqe_runner.vqe_run(max_n_iterations)
        values.append({'index': i, 'r': r, 'E': result.fun, 'n_it': result.nit})
        # t = time.time()

    values = numpy.array(values)
    pandas.DataFrame(values).to_csv('fix_ansatz_h2.csv')

    ansatz_elements = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()

    values = []
    for i in range(50):
        r = 0.4 + i * 0.02
        print('index ', i, 'R = ', r)
        vqe_runner = VQERunner(molecule, backend=QiskitSimulation, excitation_list=ansatz_elements,
                               molecule_geometry_params={'distance': r})

        result = vqe_runner.vqe_run(max_n_iterations)
        values.append({'index': i, 'r': r, 'E': result.fun, 'n_it': result.nit})

    values = numpy.array(values)
    pandas.DataFrame(values).to_csv('uccsd_ansatz_h2.csv')

    # print(result)
    # print('TIme for running: ', t - t0)

    print('Pizza')
