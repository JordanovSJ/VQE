from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner

from src.utils import QasmUtils, MatrixUtils

import itertools
import numpy

# We implement two different types of ansatz states
# Type "excitation_list" consists of a list of exponent Pauli terms representing single step Trotter excitations
# Type "qasm_list" consists of custom circuit elements represented explicitly by qasm instructions


class AnsatzElement:
    def __init__(self, element_type, element, n_var_parameters=1, excitation_order=None, fermi_operator=None):
        self.element = element
        self.element_type = element_type  # excitation or not
        self.n_var_parameters = n_var_parameters
        self.fermi_operator = fermi_operator

        if (self.element_type == 'excitation') and (excitation_order is None):
            assert type(self.element) == QubitOperator
            assert n_var_parameters == 1
            self.excitation_order = self.get_excitation_order()
        else:
            self.excitation_order = excitation_order

    def get_qasm(self, var_parameters):
        if self.element_type == 'excitation':
            assert len(var_parameters) == 1
            return QasmUtils.get_excitation_qasm(self.element, var_parameters[0])
        else:
            return self.element.format(*var_parameters)

    def get_excitation_order(self):
        terms = list(self.element)
        n_terms = len(terms)
        return max([len(terms[i]) for i in range(n_terms)])


# TODO change to static classes and methods?
class UCCSD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

    def get_single_excitation_list(self):
        single_excitations = []
        for i in range(self.n_electrons):
            for j in range(self.n_electrons, self.n_orbitals):
                fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(j, i))
                excitation = jordan_wigner(fermi_operator)
                single_excitations.append(AnsatzElement('excitation', excitation, fermi_operator=fermi_operator,
                                                        excitation_order=1))
        return single_excitations

    def get_double_excitation_list(self):
        double_excitations = []
        for i in range(self.n_electrons-1):
            for j in range(i+1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals-1):
                    for l in range(k+1, self.n_orbitals):
                        fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'.format(i, j, k, l))
                        excitation = jordan_wigner(fermi_operator)
                        double_excitations.append(AnsatzElement('excitation', excitation, fermi_operator=fermi_operator,
                                                                excitation_order=2))
        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitation_list() + self.get_double_excitation_list()


class UCCGSD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

    def get_single_excitation_list(self):
        single_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 2):
            fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(* indices))
            excitation = jordan_wigner(FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(* indices)))
            single_excitations.append(AnsatzElement('excitation', excitation, fermi_operator=fermi_operator,
                                                    excitation_order=1))
        return single_excitations

    def get_double_excitation_list(self):
        double_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 4):
            fermi_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(* indices))
            excitation = jordan_wigner(fermi_operator)
            double_excitations.append(AnsatzElement('excitation', excitation, fermi_operator=fermi_operator,
                                                    excitation_order=2))
        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitation_list() + self.get_double_excitation_list()


# this is ugly ansatz
# TODO
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
        qasm_list = [self.get_single_block(index) for index in range(1, self.n_orbitals)]
        qasm = ''.join(qasm_list)
        return [AnsatzElement('qasm', qasm, self.n_orbitals, n_var_parameters=2*self.n_orbitals*(self.n_orbitals-1))]


