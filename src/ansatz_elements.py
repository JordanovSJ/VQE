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
            var_parameters = numpy.array(var_parameters)
            return self.element.format(*var_parameters)

    def get_excitation_order(self):
        terms = list(self.element)
        n_terms = len(terms)
        return max([len(terms[i]) for i in range(n_terms)])


# Heuristic exchange ansatz 1,  17.02.2020
class ExchangeAnsatz1(AnsatzElement):
    def __init__(self, n_orbitals, n_electrons, n_blocks=1):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.n_blocks = n_blocks

        n_var_parameters = min(n_electrons, n_orbitals - n_electrons)*(1 + n_blocks)
        super(ExchangeAnsatz1, self).\
            __init__(element=None, element_type='hardware_efficient', n_var_parameters=n_var_parameters)

    def get_qasm(self, var_parameters, ):
        assert len(var_parameters) == self.n_var_parameters
        var_parameters_cycle = itertools.cycle(var_parameters)
        qasm = ['']
        for block in range(self.n_blocks):
            unoccupied_orbitals = list(range(self.n_electrons, self.n_orbitals))
            for occupied_orbital in reversed(range(0, self.n_electrons)):
                if len(unoccupied_orbitals) == 0:
                    break
                if occupied_orbital == self.n_electrons - 1:
                    virtual_orbital = self.n_electrons + block
                else:
                    virtual_orbital = min(unoccupied_orbitals)
                unoccupied_orbitals.remove(virtual_orbital)

                # add a phase rotation for the excited orbitals only
                angle = var_parameters_cycle.__next__()
                qasm.append('rz({}) q[{}];\n'.format(angle, virtual_orbital))

                angle = var_parameters_cycle.__next__()
                qasm.append(QasmUtils.get_partial_exchange_qasm(angle, occupied_orbital, virtual_orbital))

            # TODO add exchanges between the last unoccupied orbitals?

        return ''.join(qasm)


# Heuristic exchange ansatz 1,  17.02.2020
class ExchangeAnsatz2(AnsatzElement):
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

        n_var_parameters = 2*n_orbitals
        super(ExchangeAnsatz2, self).\
            __init__(element=None, element_type='hardware_efficient', n_var_parameters=n_var_parameters)

    def get_qasm(self, var_parameters):
        var_parameters *= 10
        assert len(var_parameters) == self.n_var_parameters
        qasm_even = ['']
        qasm_odd = ['']

        parameter_id = 0
        for qubit in range(self.n_orbitals ):

            if qubit % 2:
                angle = var_parameters[parameter_id]
                parameter_id += 1
                qasm_even.append('rz({}) q[{}];\n'.format(angle, qubit))

                angle = var_parameters[parameter_id]
                parameter_id += 1
                qasm_even.append('rz({}) q[{}];\n'.format(angle, (qubit+1) % self.n_orbitals))

                angle = var_parameters[parameter_id]
                parameter_id += 1
                qasm_even.append(QasmUtils.get_partial_exchange_qasm(angle, qubit, (qubit + 1) % self.n_orbitals))
            else:
                angle = var_parameters[parameter_id]
                parameter_id += 1
                qasm_odd.append(QasmUtils.get_partial_exchange_qasm(angle, qubit, (qubit + 1) % self.n_orbitals))

        assert parameter_id == len(var_parameters)
        return ''.join(qasm_odd) + ''.join(qasm_even)


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
            excitation = jordan_wigner(fermi_operator)
            single_excitations.append(AnsatzElement('excitation', excitation, fermi_operator=fermi_operator,
                                                    excitation_order=1))
        return single_excitations

    def get_double_excitation_list(self):
        double_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 4):
            fermi_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'.format(* indices))
            excitation = jordan_wigner(fermi_operator)
            double_excitations.append(AnsatzElement('excitation', excitation, fermi_operator=fermi_operator,
                                                    excitation_order=2))
        return double_excitations

    def get_ansatz_elements(self):
        return self.get_single_excitation_list() + self.get_double_excitation_list()


