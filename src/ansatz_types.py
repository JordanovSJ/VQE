from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner
import itertools
import numpy

# We implement two different types of ansatz states
# Type "excitation_list" consists of a list of exponent Pauli terms representing single step Trotter excitations
# Type "qasm_list" consists of custom circuit elements represented explicitly by qasm instructions


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


