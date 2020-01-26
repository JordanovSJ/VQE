from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner
import itertools
import numpy

# We implement two different types of ansatz states
# Type "excitation_list" consists of a list of exponent Pauli terms representing single step Trotter excitations
# Type "qasm_list" consists of custom circuit elements represented explicitly by qasm instructions


class AnsatzElement:
    def __int__(self, type, element, n_qubits):
        self.type = type  # excitation or not
        self.n_qubits = n_qubits
        # TODO
        self.qasm = None
        self.gate_counter = None
        if type == 'excitation':
            self.excitation_order = 'TODO'
        else:
            self.excitation_order = None

        def get_exponent_qasm(exponent_term, exponent_angle, gate_counter):
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

                    gate_counter['q{}'.format(qubit)]['u1'] += 2
                if pauli_operator == 'Y':
                    y_basis_correction_front.append('rx({}) q[{}];\n'.format(numpy.pi / 2, qubit))
                    y_basis_correction_back.append('rx({}) q[{}];\n'.format(- numpy.pi / 2, qubit))

                    gate_counter['q{}'.format(qubit)]['u1'] += 2

                # add the core cnot gates
                if i > 0:
                    previous_qubit = exponent_term[i - 1][0]
                    cnots.append('cx q[{}],q[{}];\n'.format(previous_qubit, qubit))

                    gate_counter['q{}'.format(previous_qubit)]['cx'] += 2
                    gate_counter['q{}'.format(qubit)]['cx'] += 2

            front_basis_correction = x_basis_correction + y_basis_correction_front
            back_basis_correction = x_basis_correction + y_basis_correction_back

            # TODO make this more readable
            # add a Z-rotation between the two CNOT ladders at the last qubit
            last_qubit = exponent_term[-1][0]
            z_rotation = 'rz({}) q[{}];\n'.format(-2 * exponent_angle, last_qubit)  # exp(i*theta*Z) ~ Rz(-2*theta)

            gate_counter['q{}'.format(last_qubit)]['u1'] += 1

            # create the cnot module simulating a single Trotter step
            cnots_module = cnots + [z_rotation] + cnots[::-1]

            return ''.join(front_basis_correction + cnots_module + back_basis_correction)



# TODO change to static classes and methods?
class UCCSD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.ansatz_type = 'excitation_list'

    def get_single_excitation_list(self):
        single_excitations = []
        for i in range(self.n_electrons):
            for j in range(self.n_electrons, self.n_orbitals):
                fermion_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(j, i))
                single_excitations.append(jordan_wigner(fermion_operator))

        # returns list of QubitOperators
        return single_excitations

    def get_double_excitation_list(self):
        double_excitations = []
        for i in range(self.n_electrons-1):
            for j in range(i+1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals-1):
                    for l in range(k+1, self.n_orbitals):
                        fermion_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                                           .format(i, j, k, l))
                        double_excitations.append(jordan_wigner(fermion_operator))

        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitation_list() + self.get_double_excitation_list(), self.ansatz_type


class UCCGSD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.ansatz_type = 'excitation_list'

    def get_single_excitation_list(self):
        single_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 2):
            fermion_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(* indices))
            single_excitations.append(jordan_wigner(fermion_operator))

        return single_excitations

    def get_double_excitation_list(self):
        double_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 4):
            fermion_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(* indices))
            double_excitations.append(jordan_wigner(fermion_operator))

        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitation_list() + self.get_double_excitation_list(), self.ansatz_type


class FixedAnsatz1:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.ansatz_type = 'qasm_list'

    def get_single_block(self, index):
        qasm = ['']
        # apply single qubit general rotations to each qubit
        for qubit in range(self.n_orbitals):
            qasm.append('rx({{}}) q[{}];\n'.format(qubit))  # we want to leave first {} empty for var_parameter later
            qasm.append('ry({{}}) q[{}];\n'.format(qubit))

        # used_qubits = numpy.zeros(self.n_orbitals)

        for qubit in range(1, self.n_orbitals):

            qasm.append('cx q[{}], q[{}];\n'.format(qubit - 1, qubit))

            # used_qubits[qubit] = 1
            # used_qubits[next_qubit] = 1

        return ''.join(qasm)

    def get_ansatz_elements(self):
        # return block
        return [self.get_single_block(index) for index in range(1, self.n_orbitals)], self.ansatz_type


