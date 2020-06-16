from src import q_systems
from src.backends import QiskitSimulation, MatrixCalculation
from src.ansatz_element_lists import UCCSD

import openfermionpsi4
import openfermion

import scipy
import unittest
import qiskit
import numpy
from src.utils import QasmUtils, MatrixUtils


class QiskitSimulationTest(unittest.TestCase):
    # test that a circuit of pauli gates only produces a correct statevector
    def test_pauli_gates_circuit_statevector(self):

        qubit_operator = openfermion.QubitOperator('X0 Y1')
        qasm_circuit = QasmUtils.qasm_header(2)
        qasm_circuit += QasmUtils.pauli_operator_qasm(qubit_operator)
        statevector = QiskitSimulation.get_statevector_from_qasm(qasm_circuit)

        expected_statevector = numpy.array([0, 0, 0, 1j])

        self.assertEqual(len(expected_statevector), len(statevector))

        for i in range(len(statevector)):
            self.assertEqual(statevector[i], expected_statevector[i])

    # test that a circuit of for an exponent of pauli gates produces a correct statevector
    def test_exponent_statevector(self):

        exp_operator = ((0, 'X'), (1, 'Z'), (2, 'Z'))
        qasm = QasmUtils.qasm_header(3)
        qasm += QasmUtils.exponent_qasm(exp_operator, -numpy.pi / 2)
        statevector = QiskitSimulation.get_statevector_from_qasm(qasm)

        expected_statevector = numpy.zeros(8)
        expected_statevector[1] = 1

        self.assertEqual(len(expected_statevector), len(statevector))

        for i in range(len(statevector)):
            self.assertEqual(statevector[i].round(3), expected_statevector[i].round(3))

    # test that the the qiskit circuit simulator and a matrices calculation produce same statevector for an exponent
    # TODO: this test is ugly ...
    def test_exponent_statevectors(self):

        qubit_operators = []
        # only symetric qubit operators will works, because qiskit and openfermion use different qubit orderings
        qubit_operators.append(openfermion.QubitOperator('Z0 Y1 Z2'))
        qubit_operators.append(openfermion.QubitOperator('X0 Y1 X2'))
        qubit_operators.append(openfermion.QubitOperator('Y0 X1 X2 Y3'))

        for qubit_operator in qubit_operators:

            qubit_operator_tuple = list(qubit_operator.terms.keys())[0]
            n_qubits = len(qubit_operator_tuple)

            for angle in range(10):
                angle = 2*numpy.pi/10

                # <<< create a statevector using QiskitSimulation.get_exponent_qasm >>>
                qasm = QasmUtils.qasm_header(n_qubits)
                qasm += QasmUtils.exponent_qasm(qubit_operator_tuple, angle)
                qiskit_statevector = QiskitSimulation.get_statevector_from_qasm(qasm)
                qiskit_statevector = qiskit_statevector * numpy.exp(1j * angle)  # correct for a global phase
                qiskit_statevector = qiskit_statevector.round(2)  # round for the purpose of testing

                # <<< create a statevector using MatrixCalculation.get_qubit_operator_exponent_matrix >>>
                exp_matrix = MatrixUtils.get_qubit_operator_exponent_matrix(1j*qubit_operator, n_qubits, angle).todense()
                # prepare initial statevector corresponding to state |0>
                array_statevector = numpy.zeros(2**n_qubits)
                array_statevector[0] = 1
                # update statevector
                array_statevector = numpy.array(exp_matrix.dot(array_statevector))[0].round(2) # round for the purpose of testing

                # <<<< compare both state vectors >>>>
                self.assertEqual(len(array_statevector), len(qiskit_statevector))
                # check the components of the two vectors are equal
                for i in range(len(qiskit_statevector)):
                    self.assertEqual(qiskit_statevector[i], array_statevector[i])

    # test that the qiskit and the matrix backends produce same value for <H> for the same given excitation parameters
    def test_energies(self):
        molecule = q_systems.H2
        molecule_data = openfermion.hamiltonians.MolecularData(geometry=molecule.get_geometry({'distance': 0.735}),
                                                               basis='sto-3g', multiplicity=molecule.multiplicity,
                                                               charge=molecule.charge)

        molecule_psi4 = openfermionpsi4.run_psi4(molecule_data)

        # Get a qubit representation of the molecule hamiltonian
        molecule_ham = molecule_psi4.get_molecular_hamiltonian()
        fermion_ham = openfermion.transforms.get_fermion_operator(molecule_ham)
        h = openfermion.transforms.jordan_wigner(fermion_ham)

        ansatz_elements = UCCSD(molecule.n_orbitals, molecule.n_electrons).get_ansatz_elements()
        var_parameters = numpy.zeros(len(ansatz_elements))
        var_parameters[-1] = 0.11
        energy_qiskit_sim = QiskitSimulation.get_expectation_value(h, ansatz_elements, var_parameters, molecule.n_orbitals,
                                                                   molecule.n_electrons)[0].real
        energy_matrix_mult = MatrixCalculation.get_energy(h, ansatz_elements, var_parameters, molecule.n_orbitals,
                                                          molecule.n_electrons)[0].real

        self.assertEqual(round(energy_qiskit_sim, 3), round(energy_matrix_mult, 3))

    # test that the qiskit and the matrix backends produce the same HF statevector
    def test_hf_states(self):
        n_qubits = 5
        n_electrons = 3

        qasm = QasmUtils.qasm_header(n_qubits)
        qasm += QasmUtils.hf_state(n_electrons)
        qasm += QasmUtils.reverse_qubits_qasm(n_qubits)
        qiskit_statevector = QiskitSimulation.get_statevector_from_qasm(qasm)

        sparse_statevector = scipy.sparse.csr_matrix(openfermion.utils.jw_hartree_fock_state(n_electrons, n_qubits))
        array_statevector = numpy.array(sparse_statevector.todense())[0]

        for i in range(len(qiskit_statevector)):
            self.assertEqual(qiskit_statevector[i], array_statevector[i])


if __name__ == '__main__':
    unittest.main()
