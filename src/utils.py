from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermion import QubitOperator
from openfermion.utils import jw_hartree_fock_state
import time

import qiskit
import scipy
import numpy
import logging
import datetime
import ray

from src import config


class QasmUtils:

    @ staticmethod
    def gate_count(qasm, n_qubits):
        gate_counter = {}
        for i in range(n_qubits):
            # count all occurrences of a qubit (can get a few more because of the header)
            qubit_count = qasm.count('q[{}]'.format(i))
            cnot_count = qasm.count('q[{}],'.format(i))
            cnot_count += qasm.count(',q[{}]'.format(i))
            cnot_count += qasm.count('q[{}] ,'.format(i))
            cnot_count += qasm.count(', q[{}]'.format(i))
            gate_counter['q{}'.format(i)] = {'cx': cnot_count, 'u1': qubit_count-cnot_count}
        return gate_counter

    @staticmethod
    def qasm_header(n_qubits):
        return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{0}];\ncreg c[{0}];\n'.format(n_qubits)

    # NOT USED
    @staticmethod
    def pauli_operator_qasm(qubit_operator):
        assert type(qubit_operator) == QubitOperator
        assert len(qubit_operator.terms) == 1

        operator = next(iter(qubit_operator.terms.keys()))
        coeff = next(iter(qubit_operator.terms.values()))
        assert coeff == 1

        # represent the qasm as a list of strings
        qasm = ['']

        for gate in operator:
            qubit = gate[0]
            if gate[1] == 'X':
                qasm.append('x q[{0}];\n'.format(qubit))
            elif gate[1] == 'Y':
                qasm.append('y q[{0}];\n'.format(qubit))
            elif gate[1] == 'Z':
                qasm.append('z q[{0}];\n'.format(qubit))
            else:
                raise ValueError('Invalid qubit operator. {} is not a Pauli operator'.format(gate[1]))

        return ''.join(qasm)

    @staticmethod
    def controlled_y_rotation(angle, control, target):
        qasm = ['']
        qasm.append('ry({}) q[{}];\n'.format(angle/2, target))
        qasm.append('cx q[{}], q[{}];\n'.format(control, target))
        qasm.append('ry({}) q[{}];\n'.format(-angle/2, target))
        qasm.append('cx q[{}], q[{}];\n'.format(control, target))

        return ''.join(qasm)

    @staticmethod
    def n_controlled_y_rotation(angle, controls, target):
        if not controls:
            return 'ry({}) q[{}];\n'.format(angle, target)
        else:
            qasm = ['']

            qasm.append(QasmUtils.n_controlled_y_rotation(angle / 2, controls[:-1], target))
            qasm.append('cx q[{}], q[{}];\n'.format(controls[-1], target))
            qasm.append(QasmUtils.n_controlled_y_rotation(-angle / 2, controls[:-1], target))
            qasm.append('cx q[{}], q[{}];\n'.format(controls[-1], target))

            return ''.join(qasm)

    @staticmethod
    def controlled_xz(qubit_1, qubit_2, reverse=False):
        qasm = ['']
        qasm.append('h q[{}];\n'.format(qubit_2))
        if reverse:
            qasm.append('rz({}) q[{}];\n'.format(numpy.pi/2, qubit_2))
            qasm.append('rz({}) q[{}];\n'.format(-numpy.pi / 2, qubit_1))
            qasm.append('cx q[{}], q[{}];\n'.format(qubit_1, qubit_2))
            qasm.append('rz({}) q[{}];\n'.format(-numpy.pi/2, qubit_2))
        else:
            qasm.append('rz({}) q[{}];\n'.format(numpy.pi / 2, qubit_2))
            qasm.append('cx q[{}], q[{}];\n'.format(qubit_1, qubit_2))
            qasm.append('rz({}) q[{}];\n'.format(-numpy.pi / 2, qubit_2))
            qasm.append('rz({}) q[{}];\n'.format(numpy.pi / 2, qubit_1))
        qasm.append('h q[{}];\n'.format(qubit_2))

        return ''.join(qasm)

    # this is simplified single particle exchange gate, that does not change phases
    @staticmethod
    def partial_exchange(angle, qubit_1, qubit_2):
        theta = numpy.pi/2 - angle
        qasm = ['']
        qasm.append(QasmUtils.controlled_xz(qubit_2, qubit_1))

        qasm.append('ry({}) q[{}];\n'.format(theta, qubit_2))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_1, qubit_2))
        qasm.append('ry({}) q[{}];\n'.format(-theta, qubit_2))

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_2, qubit_1))

        return ''.join(qasm)

    # return a qasm circuit for preparing the HF state
    @staticmethod
    def hf_state(n_electrons):
        qasm = ['']
        for i in range(n_electrons):
            qasm.append('x q[{0}];\n'.format(i))

        return ''.join(qasm)

    # get the qasm circuit of an excitation
    @staticmethod
    def fermi_excitation_qasm(excitation, var_parameter):
        qasm = ['']
        for exponent_term in excitation.terms:
            exponent_angle = var_parameter * excitation.terms[exponent_term]
            assert exponent_angle.real == 0
            exponent_angle = exponent_angle.imag
            qasm.append(QasmUtils.exponent_qasm(exponent_term, exponent_angle))

        return ''.join(qasm)

    # returns a qasm circuit for an exponent of pauli operators
    @staticmethod
    def exponent_qasm(exponent_term, exponent_angle):
        assert type(exponent_term) == tuple  # TODO remove?
        assert exponent_angle.imag == 0

        # gates for X and Y basis correction (Z by default)
        x_basis_correction = ['']
        y_basis_correction_front = ['']
        y_basis_correction_back = ['']
        # CNOT ladder
        cnots = ['']

        for i, operator in enumerate(exponent_term):
            qubit = operator[0]
            pauli_operator = operator[1]

            # add basis rotations for X and Y
            if pauli_operator == 'X':
                x_basis_correction.append('h q[{}];\n'.format(qubit))

            if pauli_operator == 'Y':
                y_basis_correction_front.append('rx({}) q[{}];\n'.format(numpy.pi / 2, qubit))
                y_basis_correction_back.append('rx({}) q[{}];\n'.format(- numpy.pi / 2, qubit))

            # add the core cnot gates
            if i > 0:
                previous_qubit = exponent_term[i - 1][0]
                cnots.append('cx q[{}],q[{}];\n'.format(previous_qubit, qubit))

        front_basis_correction = x_basis_correction + y_basis_correction_front
        back_basis_correction = x_basis_correction + y_basis_correction_back

        # TODO make this more readable
        # add a Z-rotation between the two CNOT ladders at the last qubit
        last_qubit = exponent_term[-1][0]
        z_rotation = 'rz({}) q[{}];\n'.format(-2*exponent_angle, last_qubit)  # exp(i*theta*Z) ~ Rz(-2*theta)

        # create the cnot module simulating a single Trotter step
        cnots_module = cnots + [z_rotation] + cnots[::-1]

        return ''.join(front_basis_correction + cnots_module + back_basis_correction)

    # get a circuit of SWAPs to reverse the order of qubits
    @staticmethod
    def reverse_qubits_qasm(n_qubits):
        qasm = ['']
        for i in range(int(n_qubits/2)):
            qasm.append('swap q[{}], q[{}];\n'.format(i, n_qubits - i - 1))

        return ''.join(qasm)


class MatrixUtils:
    # NOT USED
    @staticmethod
    def get_statevector_module(sparse_statevector):
        return numpy.sqrt(sparse_statevector.conj().dot(sparse_statevector.transpose()).todense().item())

    # NOT USED
    @staticmethod
    def renormalize_statevector(sparse_statevector):
        statevector_module = numpy.sqrt(sparse_statevector.conj().dot(sparse_statevector.transpose()).todense().item())
        assert statevector_module.imag == 0
        return sparse_statevector / statevector_module

    # returns the compressed sparse row matrix for the exponent of a qubit operator
    @staticmethod
    def get_qubit_operator_exponent_matrix(qubit_operator, n_qubits, parameter=1):
        assert parameter.imag == 0  # TODO remove?
        qubit_operator_matrix = get_sparse_operator(qubit_operator, n_qubits)
        return scipy.sparse.linalg.expm(parameter * qubit_operator_matrix)


class AdaptAnsatzUtils:
    # finds the VQE energy for a single ansatz element added to (optionally) an initial ansatz
    @staticmethod
    def get_ansatz_elements_vqe_results(vqe_runner, ansatz_elements, initial_var_parameters=None,
                                        initial_ansatz=None, multithread=False):
        if initial_ansatz is None:
            initial_ansatz = []
        if multithread:
            ray.init(num_cpus=config.multithread['n_cpus'])
            elements_ray_ids = [
                [element,
                 vqe_runner.vqe_run_multithread.remote(self=vqe_runner, ansatz_elements=initial_ansatz + [element],
                                                       initial_var_parameters=initial_var_parameters)]
                for element in ansatz_elements
            ]
            elements_results = [[element_ray_id[0], ray.get(element_ray_id[1])] for element_ray_id in elements_ray_ids]
            ray.shutdown()
        else:
            elements_results = [
                [element, vqe_runner.vqe_run(ansatz_elements=initial_ansatz + [element],
                                             initial_var_parameters=initial_var_parameters)]
                for element in ansatz_elements
            ]

        return elements_results

    # returns the ansatz element that achieves lowest energy (together with the energy value)
    @staticmethod
    def get_most_significant_ansatz_element(vqe_runner, ansatz_elements, initial_var_parameters=None,
                                            initial_ansatz=None, multithread=False):
        elements_results = AdaptAnsatzUtils.get_ansatz_elements_vqe_results(vqe_runner, ansatz_elements,
                                                                            initial_var_parameters=initial_var_parameters,
                                                                            initial_ansatz=initial_ansatz,
                                                                            multithread=multithread)
        return min(elements_results, key=lambda x: x[1].fun)

    # get ansatz elements that contribute to energy decrease below(above) some threshold value
    @staticmethod
    def get_ansatz_elements_above_threshold(vqe_runner, ansatz_elements, threshold, initial_var_parameters=None,
                                            initial_ansatz=None, multithread=False):
        elements_results = AdaptAnsatzUtils.get_ansatz_elements_vqe_results(vqe_runner, ansatz_elements,
                                                                            initial_var_parameters=initial_var_parameters,
                                                                            initial_ansatz=initial_ansatz,
                                                                            multithread=multithread)
        return [element_result for element_result in elements_results if element_result[1].fun <= threshold]


class LogUtils:

    @staticmethod
    def log_cofig():
        time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
        logging_filename = '{}'.format(time_stamp)
        try:
            logging.basicConfig(filename='../../results/logs/{}.txt'.format(logging_filename), level=logging.INFO,
                                format='%(levelname)s %(asctime)s %(message)s')
        except FileNotFoundError:
            try:
                logging.basicConfig(filename='../results/logs/{}.txt'.format(logging_filename), level=logging.INFO,
                                    format='%(levelname)s %(asctime)s %(message)s')
            except FileNotFoundError:
                logging.basicConfig(filename='results/logs/{}.txt'.format(logging_filename), level=logging.INFO,
                                    format='%(levelname)s %(asctime)s %(message)s')

        # disable logging from qiskit
        logging.getLogger('qiskit').setLevel(logging.WARNING)

    @staticmethod
    def vqe_info(molecule, ansatz_elements=None, basis=None, molecule_geometry_params=None, backend=None):
        logging.info('Initialize VQE for {}; n_electrons={}, n_orbitals={}; geometry: {}'
                     .format(molecule.name, molecule.n_electrons, molecule.n_orbitals, molecule_geometry_params))
        if basis is not None:
            logging.info('Basis: {}'.format(basis))
        if backend is not None:
            logging.info('Backend: {}'.format(backend))
        if ansatz_elements is not None:
            n_var_params = sum([el.n_var_parameters for el in ansatz_elements])
            logging.info('Number of Ansatz elements: {}'.format(len(ansatz_elements)))
            logging.info('Number of variational parameters: {}'.format(n_var_params))
