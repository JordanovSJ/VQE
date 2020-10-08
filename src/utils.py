from openfermion.transforms import get_sparse_operator
from openfermion import QubitOperator

import sys
import qiskit
import scipy
import numpy
import logging
import datetime


class QasmUtils:

    @staticmethod
    def ansatz_qasm(ansatz_elements, var_parameters):
        qasm = ['']
        # perform ansatz operations
        n_used_var_pars = 0
        for element in ansatz_elements:
            # take unused var. parameters for the ansatz element
            element_var_pars = var_parameters[n_used_var_pars:(n_used_var_pars + element.n_var_parameters)]
            n_used_var_pars += len(element_var_pars)
            qasm_element = element.get_qasm(element_var_pars)
            qasm.append(qasm_element)

        return ''.join(qasm)

    @staticmethod
    def gate_count_from_qasm(qasm, n_qubits):
        gate_counter = {}
        max_cnot_depth = 0
        max_u1_depth = 0
        total_cnot_count = 0
        total_u1_count = 0
        for i in range(n_qubits):
            # count all occurrences of a qubit (can get a few more because of the header)
            cnot_count = qasm.count('q[{}],'.format(i))
            cnot_count += qasm.count(',q[{}]'.format(i))
            cnot_count += qasm.count('q[{}] ,'.format(i))
            cnot_count += qasm.count(', q[{}]'.format(i))
            total_cnot_count += cnot_count
            max_cnot_depth = max(max_cnot_depth, cnot_count)

            qubit_count = qasm.count('q[{}]'.format(i))
            u1_count = qubit_count - cnot_count
            total_u1_count += u1_count
            max_u1_depth = max(max_u1_depth, u1_count)

            gate_counter['q{}'.format(i)] = {'cx': cnot_count, 'u1': u1_count}
        return {'gate_count': gate_counter, 'u1_depth': max_u1_depth, 'cnot_depth': max_cnot_depth,
                'cnot_count': total_cnot_count/2, 'u1_count': total_u1_count}

    @staticmethod
    def gate_count_from_ansatz_elements(ansatz_elements, n_qubits, var_parameters=None):
        n_var_parameters = sum([x.n_var_parameters for x in ansatz_elements])
        if var_parameters is None:
            var_parameters = numpy.zeros(n_var_parameters)
        else:
            assert n_var_parameters == len(var_parameters)
        qasm = QasmUtils.ansatz_qasm(ansatz_elements, var_parameters)
        return QasmUtils.gate_count_from_qasm(qasm, n_qubits)

    @staticmethod
    def get_circuit_matrix(qasm):
        backend = qiskit.Aer.get_backend('unitary_simulator')
        qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm)
        result = qiskit.execute(qiskit_circuit, backend).result()
        matrix = result.get_unitary(qiskit_circuit, decimals=5)
        return matrix

    @staticmethod
    def qasm_header(n_qubits):
        return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{0}];\ncreg c[{0}];\n'.format(n_qubits)

    @staticmethod
    def pauli_word_qasm(operator):
        assert type(operator) == QubitOperator
        assert len(operator.terms) == 1
        assert next(iter(operator.terms.values())) == 1

        pauli_word = next(iter(operator.terms.keys()))

        qasm = ['']

        for gate in pauli_word:
            qubit = gate[0]
            if gate[1] == 'X':
                qasm.append('x q[{0}];\n'.format(qubit))
            elif gate[1] == 'Y':
                qasm.append('y q[{0}];\n'.format(qubit))
            elif gate[1] == 'Z':
                qasm.append('z q[{0}];\n'.format(qubit))
            else:
                raise ValueError('Invalid Pauli-word operator. {} is not a Pauli operator'.format(gate[1]))

        return ''.join(qasm)

    @staticmethod
    def controlled_y_rotation(angle, control, target):
        qasm = ['']
        qasm.append('ry({}) q[{}];\n'.format(angle/2, target))
        qasm.append('cx q[{}], q[{}];\n'.format(control, target))
        qasm.append('ry({}) q[{}];\n'.format(-angle/2, target))
        qasm.append('cx q[{}], q[{}];\n'.format(control, target))

        return ''.join(qasm)

    # equivalent single qubit excitation
    @staticmethod
    def partial_exchange(angle, qubit_1, qubit_2):
        theta = numpy.pi / 2 + angle
        qasm = ['']
        qasm.append(QasmUtils.controlled_xz(qubit_2, qubit_1))

        qasm.append('ry({}) q[{}];\n'.format(theta, qubit_2))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_1, qubit_2))
        qasm.append('ry({}) q[{}];\n'.format(-theta, qubit_2))

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_2, qubit_1))

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

    # return a qasm circuit for preparing the HF state
    @staticmethod
    def hf_state(n_electrons):
        qasm = ['']
        for i in range(n_electrons):
            qasm.append('x q[{0}];\n'.format(i))

        return ''.join(qasm)

    # get the qasm circuit of an excitation
    @staticmethod
    def fermi_excitation(excitation, var_parameter):
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
    def get_excitation_matrix(excitation_operator, n_qubits, parameter=1):
        assert parameter.imag == 0  # TODO remove?
        qubit_operator_matrix = get_sparse_operator(excitation_operator, n_qubits)
        return scipy.sparse.linalg.expm(parameter * qubit_operator_matrix)


class LogUtils:

    @staticmethod
    def log_config():
        time_stamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
        logging_filename = '{}'.format(time_stamp)
        stdout_handler = logging.StreamHandler(sys.stdout)

        try:
            logging.basicConfig(filename='../../results/logs/{}.txt'.format(logging_filename), level=logging.DEBUG,
                                format='%(levelname)s %(asctime)s %(message)s')
        except FileNotFoundError:
            try:
                logging.basicConfig(filename='../results/logs/{}.txt'.format(logging_filename), level=logging.DEBUG,
                                    format='%(levelname)s %(asctime)s %(message)s')
            except FileNotFoundError:
                logging.basicConfig(filename='results/logs/{}.txt'.format(logging_filename), level=logging.DEBUG,
                                    format='%(levelname)s %(asctime)s %(message)s')

        # make logger print to console (it will not if multithreaded)
        logging.getLogger().addHandler(stdout_handler)

        # disable logging from qiskit
        logging.getLogger('qiskit').setLevel(logging.WARNING)

    @staticmethod
    def vqe_info(molecule, ansatz=None, basis=None, molecule_geometry_params=None, backend=None):
        logging.info('Initialize VQE for {}; n_electrons={}, n_orbitals={}; geometry: {}'
                     .format(molecule.name, molecule.n_electrons, molecule.n_orbitals, molecule_geometry_params))
        if basis is not None:
            logging.info('Basis: {}'.format(basis))
        if backend is not None:
            logging.info('Backend: {}'.format(backend))
        if ansatz is not None:
            n_var_params = sum([el.n_var_parameters for el in ansatz])
            logging.info('Number of Ansatz elements: {}'.format(len(ansatz)))
            logging.info('Number of variational parameters: {}'.format(n_var_params))
