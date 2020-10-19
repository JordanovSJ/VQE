import logging

from src.iter_vqe_utils import *


class Cache:
    def __init__(self, H_sparse_matrix, n_qubits, n_electrons, exc_gen_matrices=None):
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.statevector = None
        self.var_parameters = None
        self.H_sparse_matrix = H_sparse_matrix
        self.exc_gen_matrices = exc_gen_matrices

    def update_statevector(self, ansatz, var_parameters, init_state_qasm=None):
        # TODO add single parameter functionnality with init_statevector
        assert len(var_parameters) == len(ansatz)
        if self.var_parameters is not None and var_parameters == self.var_parameters:  # this condition is not neccesserily sufficinet
            assert self.statevector is not None
        else:
            self.var_parameters = var_parameters
            self.statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, self.n_qubits,
                                                                 self.n_electrons, init_state_qasm=init_state_qasm)
        return self.statevector


class GlobalCache(Cache):
    def __init__(self, q_system, excited_state=0):
        self.q_system = q_system
        self.commutator_matrices = None
        # self.exc_gen_matrices = None
        H_sparse_matrix = get_sparse_operator(q_system.jw_qubit_ham)
        if excited_state > 0:
            H_sparse_matrix = QiskitSim. ham_sparse_matrix(q_system, excited_state=excited_state)
            if H_sparse_matrix.data.nbytes > config.matrix_size_threshold:
                # decrease the size of the matrix. Typically it will have a lot of insignificant very small (~1e-19)
                # elements that do not contribute to the accuracy but inflate the size of the matrix (~200 MB for Lih)
                H_sparse_matrix = scipy.sparse.csr_matrix(H_sparse_matrix.todense().round(config.floating_point_accuracy_digits))

        super(GlobalCache, self).__init__(H_sparse_matrix=H_sparse_matrix, n_qubits=q_system.n_qubits,
                                          n_electrons=q_system.n_electrons)

    def get_grad_thread_cache(self, ansatz_element, sparse_statevector):
        # TODO check if copy is neccessery
        thread_cache = ThreadCache(commutator_sparse_matrix=self.commutator_matrices[str(ansatz_element.excitation_generator)].copy(),
                                   init_sparse_statevector=sparse_statevector.copy(), n_qubits=self.q_system.n_qubits,
                                   n_electrons=self.q_system.n_electrons)
        return thread_cache

    def get_vqe_thread_cache(self):
        # TODO check if copy is neccessery
        thread_cache = ThreadCache(H_sparse_matrix=self.H_sparse_matrix.copy(),
                                   exc_gen_matrices=self.get_exc_gen_matrices_copy(),
                                   n_qubits=self.q_system.n_qubits, n_electrons=self.q_system.n_electrons)
        return thread_cache

    def single_par_vqe_thread_cache(self, ansatz_element, init_statevector):
        # TODO check if copy is neccessery
        exc_gen = ansatz_element.excitation_generator
        exc_gen_matrix = self.exc_gen_matrices[str(exc_gen)]
        thread_cache = ThreadCache(H_sparse_matrix=self.H_sparse_matrix.copy(),
                                   commutator_sparse_matrix=self.commutator_matrices[str(exc_gen)].copy(),
                                   init_sparse_statevector=init_statevector.copy(),
                                   n_qubits=self.q_system.n_qubits, n_electrons=self.q_system.n_electrons,
                                   exc_gen_matrices={str(exc_gen): exc_gen_matrix.copy()})
        return thread_cache

    def get_exc_gen_matrices_copy(self):
        exc_gen_matrices_copy = {}
        for exc_gen in self.exc_gen_matrices.keys():
            exc_gen_matrices_copy[str(exc_gen)] = self.exc_gen_matrices[str(exc_gen)].copy()
        return exc_gen_matrices_copy

    def calculate_exc_gen_matrices(self, ansatz_elements):
        logging.info('Calculating excitation generators')
        exc_generators = {}
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
                exc_generators[key] = ray.get(element_ray_id[1])

            del elements_ray_ids
            ray.shutdown()
        else:
            for i, element in enumerate(ansatz_elements):
                excitation_generator = element.excitation_generator
                key = str(excitation_generator)
                logging.info('Calculated commutator ', key)
                exc_gen_sparse_matrix = get_sparse_operator(excitation_generator, n_qubits=self.q_system.n_qubits)
                exc_generators[key] = exc_gen_sparse_matrix

        self.exc_gen_matrices = exc_generators
        return exc_generators

    def calculate_commutators_matrices(self, ansatz_elements):
        logging.info('Calculating commutators')

        if self.exc_gen_matrices is None:
            self.calculate_exc_gen_matrices(ansatz_elements)

        commutators = {}
        if config.multithread:
            ray.init(num_cpus=config.ray_options['n_cpus'])
            elements_ray_ids = [
                [
                    element, GlobalCache.get_commutator_matrix_multithread.
                    remote(self.exc_gen_matrices[str(element.excitation_generator)].copy(), self.H_sparse_matrix.copy())
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
                logging.info('Calculated commutator ', key)
                exc_gen_sparse_matrix = self.exc_gen_matrices[key]
                commutator_sparse_matrix = self.H_sparse_matrix * exc_gen_sparse_matrix - exc_gen_sparse_matrix * self.H_sparse_matrix
                commutators[key] = commutator_sparse_matrix

        self.commutator_matrices = commutators
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
        print('Calculated excitation matrix time ', time.time() - t0)
        return exc_gen_matrix


# TODO make a subclass of global cache?
class ThreadCache(Cache):
    def __init__(self, n_qubits, n_electrons, H_sparse_matrix=None, commutator_sparse_matrix=None,
                 init_sparse_statevector=None, exc_gen_matrices=None):
        self.commutator_sparse_matrix = commutator_sparse_matrix
        self.init_sparse_statevector = init_sparse_statevector

        super(ThreadCache, self).__init__(H_sparse_matrix=H_sparse_matrix, n_qubits=n_qubits, n_electrons=n_electrons,
                                          exc_gen_matrices=exc_gen_matrices)
    #     self.statevector = None
    #     self.var_parameters = None
    #
    # def update_statevector(self, ansatz, var_parameters, init_state_qasm=None):
    #     # TODO add single parameter functionnality with init_statevector
    #
    #     if self.var_parameters is not None and var_parameters == self.var_parameters:  # this condition is not neccesserily sufficinet
    #         assert self.statevector is not None
    #     else:
    #         if self.var_parameters is not None:
    #             assert len(self.var_parameters) == var_parameters
    #         self.var_parameters = var_parameters
    #         self.statevector = QiskitSim.statevector_from_ansatz(ansatz, var_parameters, self.n_qubits,
    #                                                              self.n_electrons, init_state_qasm=init_state_qasm)
    #     return self.statevector

