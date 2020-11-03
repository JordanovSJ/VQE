from src.backends import QiskitSim
from src import config

from openfermion import get_sparse_operator

import scipy
import ray
import numpy
import logging
import time


class Cache:
    def __init__(self, H_sparse_matrix, n_qubits, n_electrons, exc_gen_sparse_matrices=None,
                 commutator_sparse_matrices=None, sparse_statevector=None, init_sparse_statevector=None,
                 sqr_exc_gen_sparse_matrices=None):
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.sparse_statevector = sparse_statevector  # used in: normal vqe run/ excitation gradient calculations
        self.var_parameters = None  # used in: normal vqe run/ single var. parameter vqe
        self.H_sparse_matrix = H_sparse_matrix  # used in: normal vqe run/ single var. parameter vqe
        self.exc_gen_sparse_matrices = exc_gen_sparse_matrices  # used in:  single var. parameter vqe
        self.sqr_exc_gen_sparse_matrices = sqr_exc_gen_sparse_matrices
        self.commutator_sparse_matrices = commutator_sparse_matrices  # used in: single var. parameter vqe/ excitation gradient calculations
        self.init_sparse_statevector = init_sparse_statevector  # used in: single var. parameter vqe

    def hf_statevector(self):
        statevector = numpy.zeros(2 ** self.n_qubits)
        # MAGIC
        hf_term = 0
        for i in range(self.n_electrons):
            hf_term += 2 ** (self.n_qubits - 1 - i)
        statevector[hf_term] = 1
        return statevector

    # # return a statevector in the form of an array from a list of ansatz elements
    # @staticmethod
    # def statevector_from_ansatz(ansatz, var_parameters, n_qubits, n_electrons, cache, init_state_qasm=None):
    #     assert cache is not None
    #     statevector = ExcStateSim.hf_statevector(n_qubits, n_electrons)
    #     sparse_statevector = scipy.sparse.csr_matrix(statevector).transpose()
    #     identity = scipy.sparse.identity(2 ** n_qubits)
    #
    #     for i, excitation in enumerate(ansatz):
    #         parameter = var_parameters[i]
    #         term1 = numpy.sin(parameter)*cache.get_exc_gen_sparse_matrix(excitation)
    #         term2 = (1 - numpy.cos(parameter))*cache.get_sqr_exc_gen_sparse_matrix(excitation)
    #         excitation_matrix = identity + term1 + term2
    #         sparse_statevector = excitation_matrix.dot(sparse_statevector)
    #     statevector = numpy.array(sparse_statevector.transpose().todense())[0]
    #     return statevector

    def update_statevector(self, ansatz, var_parameters, init_state_qasm=None, backend=QiskitSim):
        # TODO add single parameter functionnality with init_statevector
        assert len(var_parameters) == len(ansatz)
        if self.var_parameters is not None and var_parameters == self.var_parameters:  # this condition is not neccessarily sufficient
            assert self.sparse_statevector is not None
        else:

            statevector = self.hf_statevector()
            sparse_statevector = scipy.sparse.csr_matrix(statevector).transpose().conj()
            identity = scipy.sparse.identity(2 ** self.n_qubits)

            for i, excitation in enumerate(ansatz):
                parameter = var_parameters[i]
                term1 = numpy.sin(parameter) * self.get_exc_gen_sparse_matrix(excitation)
                term2 = (1 - numpy.cos(parameter)) * self.get_sqr_exc_gen_sparse_matrix(excitation)
                excitation_matrix = identity + term1 + term2
                sparse_statevector = excitation_matrix.dot(sparse_statevector)
            # statevector = numpy.array(sparse_statevector.transpose().todense())[0]
            self.sparse_statevector = sparse_statevector.transpose().conj()

        #     # if just a single ansatz element is supplied, just add its matrix to the initial statevector
        #     if self.init_sparse_statevector is not None and len(ansatz) == 1:
        #         key = str(ansatz[0].excitation_generator)
        #         exc_gen_matrix = self.exc_gen_sparse_matrices[key]
        #         self.sparse_statevector =\
        #             scipy.sparse.linalg.expm_multiply(var_parameters[0] * exc_gen_matrix,
        #                                               self.init_sparse_statevector.transpose().conj()).transpose().conj()
        #     else:
        #         self.var_parameters = var_parameters
        #         statevector = backend.statevector_from_ansatz(ansatz, var_parameters, self.n_qubits, self.n_electrons,
        #                                                       init_state_qasm=init_state_qasm)
        #         self.sparse_statevector = scipy.sparse.csr_matrix(statevector)
        # # print(self.sparse_statevector.todense())

        print(self.sparse_statevector.dot(self.sparse_statevector.transpose().conj()).todense())

        return self.sparse_statevector

    def get_exc_gen_sparse_matrix(self, excitation):
        key = str(excitation.excitation_generator)
        return self.exc_gen_sparse_matrices[key]

    def get_sqr_exc_gen_sparse_matrix(self, excitation):
        key = str(excitation.excitation_generator)
        return self.sqr_exc_gen_sparse_matrices[key]

    def get_commutator_sparse_matrix(self, excitation):
        key = str(excitation.excitation_generator)
        return self.commutator_sparse_matrices[key]


class GlobalCache(Cache):
    def __init__(self, q_system, excited_state=0, backend=QiskitSim):
        self.q_system = q_system

        H_sparse_matrix = get_sparse_operator(q_system.jw_qubit_ham)
        if excited_state > 0:
            H_sparse_matrix = backend. ham_sparse_matrix(q_system, excited_state=excited_state)
            if H_sparse_matrix.data.nbytes > config.matrix_size_threshold:
                # decrease the size of the matrix. Typically it will have a lot of insignificant very small (~1e-19)
                # elements that do not contribute to the accuracy but inflate the size of the matrix (~200 MB for Lih)
                H_sparse_matrix = scipy.sparse.csr_matrix(H_sparse_matrix.todense().round(config.floating_point_accuracy_digits))

        super(GlobalCache, self).__init__(H_sparse_matrix=H_sparse_matrix, n_qubits=q_system.n_qubits,
                                          n_electrons=q_system.n_electrons, commutator_sparse_matrices=None)

    def get_grad_thread_cache(self, ansatz_element, sparse_statevector):
        # TODO check if copy is necessary
        key = str(ansatz_element.excitation_generator)
        commutator_matrix = self.commutator_sparse_matrices[key].copy()
        thread_cache = ThreadCache(commutator_sparse_matrices={key: commutator_matrix},
                                   sparse_statevector=sparse_statevector.copy(), n_qubits=self.q_system.n_qubits,
                                   n_electrons=self.q_system.n_electrons)
        return thread_cache

    def get_vqe_thread_cache(self):
        # TODO check if copy is necessary
        thread_cache = ThreadCache(H_sparse_matrix=self.H_sparse_matrix.copy(),
                                   exc_gen_matrices=self.get_exc_gen_matrices_copy(),
                                   # TODO
                                   # sqr_exc_gen_matrices=self.get_sqr_exc_gen_matrices_copy(),
                                   n_qubits=self.q_system.n_qubits, n_electrons=self.q_system.n_electrons)
        return thread_cache

    def single_par_vqe_thread_cache(self, ansatz_element, init_sparse_statevector):
        # TODO check if copy is necessary
        key = str(ansatz_element.excitation_generator)
        exc_gen_matrix = self.exc_gen_sparse_matrices[key].copy()
        # TODO
        # sqr_exc_gen_matrix = self.sqr_exc_gen_sparse_matrices[key].copy()

        commutator_matrix = self.commutator_sparse_matrices[key].copy()
        thread_cache = ThreadCache(H_sparse_matrix=self.H_sparse_matrix.copy(),
                                   commutator_sparse_matrices={key: commutator_matrix},
                                   init_sparse_statevector=init_sparse_statevector.copy(),
                                   n_qubits=self.q_system.n_qubits, n_electrons=self.q_system.n_electrons,
                                   exc_gen_matrices={key: exc_gen_matrix},
                                   # TODO
                                   # sqr_exc_gen_matrices={key: sqr_exc_gen_matrix}
                                   )
        return thread_cache

    def get_exc_gen_matrices_copy(self):
        exc_gen_matrices_copy = {}
        for exc_gen in self.exc_gen_sparse_matrices.keys():
            exc_gen_matrices_copy[str(exc_gen)] = self.exc_gen_sparse_matrices[str(exc_gen)].copy()
        return exc_gen_matrices_copy

    def get_sqr_exc_gen_matrices_copy(self):
        sqr_exc_gen_matrices_copy = {}
        for exc_gen in self.sqr_exc_gen_sparse_matrices.keys():
            sqr_exc_gen_matrices_copy[str(exc_gen)] = self.sqr_exc_gen_sparse_matrices[str(exc_gen)].copy()
        return sqr_exc_gen_matrices_copy

    def calculate_exc_gen_matrices(self, ansatz_elements):
        logging.info('Calculating excitation generators')
        exc_generators = {}
        sqr_exc_generators = {}
        if config.multithread:
            ray.init(num_cpus=config.ray_options['n_cpus'])
            elements_ray_ids = [
                [
                    element, GlobalCache.get_exc_generator_matrix_multithread.remote(element, n_qubits=self.q_system.n_qubits)
                ]
                for element in ansatz_elements
            ]
            for element_ray_id in elements_ray_ids:
                key = str(element_ray_id[0].excitation_generator)
                exc_generators[key] = ray.get(element_ray_id[1])[0]
                sqr_exc_generators[key] = ray.get(element_ray_id[1])[1]

            del elements_ray_ids
            ray.shutdown()
        else:
            for i, element in enumerate(ansatz_elements):
                excitation_generator = element.excitation_generator
                key = str(excitation_generator)
                logging.info('Calculated excitation generator matrix {}'.format(key))
                exc_gen_sparse_matrix = get_sparse_operator(excitation_generator, n_qubits=self.q_system.n_qubits)
                sqr_exc_gen_sparse_matrix = exc_gen_sparse_matrix*exc_gen_sparse_matrix
                exc_generators[key] = exc_gen_sparse_matrix
                sqr_exc_generators[key] = sqr_exc_gen_sparse_matrix

        self.exc_gen_sparse_matrices = exc_generators
        self.sqr_exc_gen_sparse_matrices = sqr_exc_generators
        return exc_generators

    def calculate_commutators_matrices(self, ansatz_elements):
        logging.info('Calculating commutators')

        if self.exc_gen_sparse_matrices is None:
            self.calculate_exc_gen_matrices(ansatz_elements)

        commutators = {}
        if config.multithread:
            ray.init(num_cpus=config.ray_options['n_cpus'])
            elements_ray_ids = [
                [
                    element, GlobalCache.get_commutator_matrix_multithread.
                    remote(self.exc_gen_sparse_matrices[str(element.excitation_generator)].copy(), self.H_sparse_matrix.copy())
                ]
                for element in ansatz_elements
            ]
            for element_ray_id in elements_ray_ids:
                key = str(element_ray_id[0].excitation_generator)
                commutators[key] = ray.get(element_ray_id[1])

            del elements_ray_ids
            ray.shutdown()
        else:
            for i, element in enumerate(ansatz_elements):
                excitation_generator = element.excitation_generator
                key = str(excitation_generator)
                logging.info('Calculated commutator {}'.format(key))
                exc_gen_sparse_matrix = self.exc_gen_sparse_matrices[key]
                commutator_sparse_matrix = self.H_sparse_matrix * exc_gen_sparse_matrix - exc_gen_sparse_matrix * self.H_sparse_matrix
                commutators[key] = commutator_sparse_matrix

        self.commutator_sparse_matrices = commutators
        return commutators

    @staticmethod
    @ray.remote
    def get_commutator_matrix_multithread(exc_gen_matrix, H_sparse_matrix):
        t0 = time.time()
        commutator_sparse_matrix = H_sparse_matrix * exc_gen_matrix - exc_gen_matrix * H_sparse_matrix
        del H_sparse_matrix
        del exc_gen_matrix
        print('Calculated commutator time ', time.time() - t0)
        return commutator_sparse_matrix

    @staticmethod
    @ray.remote
    def get_exc_generator_matrix_multithread(excitation, n_qubits):
        t0 = time.time()
        exc_gen_matrix = get_sparse_operator(excitation.excitation_generator, n_qubits=n_qubits)
        sqr_exc_gen_matrix = exc_gen_matrix*exc_gen_matrix
        print('Calculated excitation matrix time ', time.time() - t0)
        return exc_gen_matrix, sqr_exc_gen_matrix


# TODO make a subclass of global cache?
class ThreadCache(Cache):
    def __init__(self, n_qubits, n_electrons, H_sparse_matrix=None, commutator_sparse_matrices=None,
                 sparse_statevector=None, init_sparse_statevector=None, exc_gen_matrices=None):

        super(ThreadCache, self).__init__(H_sparse_matrix=H_sparse_matrix, n_qubits=n_qubits, n_electrons=n_electrons,
                                          exc_gen_sparse_matrices=exc_gen_matrices,
                                          commutator_sparse_matrices=commutator_sparse_matrices,
                                          sparse_statevector=sparse_statevector,
                                          init_sparse_statevector=init_sparse_statevector)


