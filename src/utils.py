# from openfermion.transforms import get_sparse_operator
from openfermion.linalg import get_sparse_operator

from openfermion import QubitOperator

import sys
import qiskit
import scipy
import numpy
import logging
import datetime


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
        assert parameter.imag == 0  # ?
        qubit_operator_matrix = get_sparse_operator(excitation_operator, n_qubits)
        return scipy.sparse.linalg.expm(parameter * qubit_operator_matrix)

    # return the hamming weight of a statevector if all its non zero terms have the same H.w. Otherwise return False
    @staticmethod
    def statevector_hamming_weight(statevector):
        hw = None
        for i, term in enumerate(statevector):
            if term != 0:
                hw_i = bin(i).count('1')
                if hw is None:
                    hw = hw_i
                elif hw != hw_i:
                    return False

        return hw


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
    def vqe_info(q_system, backend, optimizer, ansatz):
        message = ''
        message += '-----Running VQE for: {}-----\n'.format(q_system.name)
        message += '-----Number of electrons: {}-----\n'.format(q_system.n_electrons)
        message += '-----Number of orbitals: {}-----\n'.format(q_system.n_orbitals)
        message += '-----Numeber of ansatz elements: {}-----\n'.format(len(ansatz))
        if len(ansatz) == 1:
            message += '-----Ansatz type {}------\n'.format(ansatz[0].element)
        message += '-----Statevector and energy calculated using {}------\n'.format(backend)
        message += '-----Optimizer {}------\n'.format(optimizer)

        logging.info(message)


class QasmUtils:

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
    def unitary_matrix_from_qasm(qasm):
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
    def excitation_qasm(excitation_generator, var_parameter):
        qasm = ['']
        # # if type(excitations_generators) == list:
        # excitation_generator = 0*QubitOperator('')
        # for generator in excitations_generators:
        #     excitation_generator += generator
        # # else:
        # #     excitation_generator = excitations_generators

        for exponent_term in excitation_generator.terms:
            exponent_angle = var_parameter * excitation_generator.terms[exponent_term]
            assert exponent_angle.real == 0
            exponent_angle = exponent_angle.imag
            qasm.append(QasmUtils.exponent_qasm(exponent_term, exponent_angle))

        return ''.join(qasm)

    # returns a qasm circuit for an exponent of pauli operators
    @staticmethod
    def exponent_qasm(exponent_term, exponent_parameter):
        assert type(exponent_term) == tuple  # TODO remove?
        assert exponent_parameter.imag == 0

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
        z_rotation = 'rz({}) q[{}];\n'.format(-2 * exponent_parameter, last_qubit)  # exp(i*theta*Z) ~ Rz(-2*theta)

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

    # qasm for a double qubit excitation circuit
    @staticmethod
    def d_q_exc_qasm(parameter, qubit_pair_1_ref, qubit_pair_2_ref):
        # This is not required since the qubits are not ordered as for the fermi excitation
        qubit_pair_1 = qubit_pair_1_ref.copy()
        qubit_pair_2 = qubit_pair_2_ref.copy()

        parameter = parameter * 2  # for consistency with the conventional fermi excitation
        theta = parameter / 8

        qasm = ['']

        # determine the parity of the two pairs
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))

        # apply a partial swap of qubits 0 and 2, controlled by 1 and 3 ##

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))
        # # partial ccc_y operation
        qasm.append('rz({}) q[{}];\n'.format(numpy.pi / 2, qubit_pair_1[0]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[1]))  # 0 3
        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[0]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))  # 0 2
        qasm.append('h q[{}];\n'.format(qubit_pair_2[0]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[1]))  # 0 3
        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('rz({}) q[{}];\n'.format(-numpy.pi / 2, qubit_pair_1[0]))

        # ############################## partial ccc_y operation  ############ to here

        qasm.append(QasmUtils.controlled_xz(qubit_pair_1[0], qubit_pair_2[0], reverse=True))

        # correct for parity determination
        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))

        return ''.join(qasm)

    # qasm for efficient single fermionic excitation circuit
    @staticmethod
    def eff_s_f_exc_qasm(parameter, qubit_1, qubit_2):
        theta = numpy.pi / 2 + parameter
        qasm = ['']
        if qubit_2 < qubit_1:
            x = qubit_1
            qubit_1 = qubit_2
            qubit_2 = x

        parity_qubits = list(range(qubit_1 + 1, qubit_2))

        parity_cnot_ladder = ['']
        if len(parity_qubits) > 0:
            for i in range(len(parity_qubits) - 1):
                parity_cnot_ladder.append('cx q[{}], q[{}];\n'.format(parity_qubits[i], parity_qubits[i + 1]))

            qasm += parity_cnot_ladder
            # parity dependence
            qasm.append('h q[{}];\n'.format(qubit_1))
            qasm.append('cx q[{}], q[{}];\n'.format(parity_qubits[-1], qubit_1))
            qasm.append('h q[{}];\n'.format(qubit_1))

        qasm.append(QasmUtils.controlled_xz(qubit_2, qubit_1))

        qasm.append('ry({}) q[{}];\n'.format(theta, qubit_2))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_1, qubit_2))
        qasm.append('ry({}) q[{}];\n'.format(-theta, qubit_2))

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_2, qubit_1))

        if len(parity_qubits) > 0:
            qasm.append('h q[{}];\n'.format(qubit_1))
            qasm.append('cx q[{}], q[{}];\n'.format(parity_qubits[-1], qubit_1))
            qasm.append('h q[{}];\n'.format(qubit_1))

            qasm += parity_cnot_ladder[::-1]

        return ''.join(qasm)

    # qasm for efficient double fermionic excitation circuit
    @staticmethod
    def eff_d_f_exc_qasm(parameter, qubit_pair_1_ref, qubit_pair_2_ref):

        # TODO: use proper get set functions
        # use copies of the qubit pairs, in order to change the original pairs
        qubit_pair_1 = qubit_pair_1_ref.copy()
        qubit_pair_2 = qubit_pair_2_ref.copy()

        # !!!!!!!!! accounts for the missing functionality in the eff_d_f_exc circuit !!!!!
        if qubit_pair_1[0] > qubit_pair_1[1]:
            parameter *= -1
        if qubit_pair_2[0] > qubit_pair_2[1]:
            parameter *= -1

        parameter = - parameter * 2  # the factor of -2 is for consistency with the conventional fermi excitation
        theta = parameter / 8

        qasm = ['']

        qubit_pair_1.sort()
        qubit_pair_2.sort()

        all_qubits = qubit_pair_1 + qubit_pair_2
        all_qubits.sort()

        # do not include the first qubits of qubit_pair_1 and qubit_pair_2
        parity_qubits = list(range(all_qubits[0]+1, all_qubits[1])) + list(range(all_qubits[2]+1, all_qubits[3]))

        # ladder of CNOT used to determine the parity
        parity_cnot_ladder = ['']
        if len(parity_qubits) > 0:
            for i in range(len(parity_qubits) - 1):
                parity_cnot_ladder.append('cx q[{}], q[{}];\n'.format(parity_qubits[i], parity_qubits[i + 1]))
            # parity_cnot_ladder.append('x q[{}];\n'.format(parity_qubits[-1]))

        # determine the parity of the two pairs
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))

        # apply a partial swap of qubits 0 and 2, controlled by 1 and 3 ##

        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))

        # apply parity sign correction 1
        if len(parity_qubits) > 0:
            qasm += parity_cnot_ladder
            qasm.append('h q[{}];\n'.format(parity_qubits[-1]))
            qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], parity_qubits[-1]))

        # # partial ccc_y operation
        qasm.append('rz({}) q[{}];\n'.format(numpy.pi / 2, qubit_pair_1[0]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[1]))  # 0 3
        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[0]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[0]))  # 0 2
        qasm.append('h q[{}];\n'.format(qubit_pair_2[0]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_2[1]))  # 0 3
        qasm.append('h q[{}];\n'.format(qubit_pair_2[1]))

        qasm.append('rx({}) q[{}];\n'.format(theta, qubit_pair_1[0]))  # +

        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], qubit_pair_1[1]))  # 0 1
        qasm.append('h q[{}];\n'.format(qubit_pair_1[1]))

        qasm.append('rx({}) q[{}];\n'.format(-theta, qubit_pair_1[0]))  # -

        qasm.append('rz({}) q[{}];\n'.format(-numpy.pi / 2, qubit_pair_1[0]))

        # ############################## partial ccc_y operation  ############ until here

        # apply parity sign correction 2
        if len(parity_qubits) > 0:
            qasm.append('cx q[{}], q[{}];\n'.format(qubit_pair_1[0], parity_qubits[-1]))
            qasm.append('h q[{}];\n'.format(parity_qubits[-1]))
            qasm += parity_cnot_ladder[::-1]

        qasm.append(QasmUtils.controlled_xz(qubit_pair_1[0], qubit_pair_2[0], reverse=True))

        # correct for parity determination
        qasm.append('x q[{}];\n'.format(qubit_pair_1[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_1))
        qasm.append('x q[{}];\n'.format(qubit_pair_2[1]))
        qasm.append('cx q[{}], q[{}];\n'.format(*qubit_pair_2))

        return ''.join(qasm)
