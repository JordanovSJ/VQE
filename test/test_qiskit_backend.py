import unittest
import qiskit
import openfermion
from src.backends import QiskitSimulation, MatrixCalculation
import numpy


class QiskitSimulationBackendTest(unittest.TestCase):
    def test_pauli_gates_circuit_statevector(self):
        qubit_operator = openfermion.QubitOperator('X0 Y1')
        qasm_circuit = QiskitSimulation.get_qasm_header(2)
        qasm_circuit += QiskitSimulation.get_qubit_operator_qasm(qubit_operator)
        statevector = QiskitSimulation.get_statevector_from_qasm(qasm_circuit)

        expected_statevector = numpy.array([0, 0, 0, 1j])

        self.assertEqual(len(expected_statevector), len(statevector))

        for i in range(len(statevector)):
            self.assertEqual(statevector[i], expected_statevector[i])

    def test_exponent_statevector(self):
        exp_operator = ((0, 'X'), (1, 'Z'), (2, 'Z'))
        qasm = QiskitSimulation.get_qasm_header(3)
        qasm += QiskitSimulation.get_exponent_qasm(exp_operator, 1j*numpy.pi)
        statevector = QiskitSimulation.get_statevector_from_qasm(qasm)

        expected_statevector = numpy.zeros(8)
        expected_statevector[1] = 1

        self.assertEqual(len(expected_statevector), len(statevector))

        for i in range(len(statevector)):
            self.assertEqual(statevector[i], expected_statevector[i])

    def test_exponent_statevector_with_matrix_backend(self):
        exp_operator_tuple = ((0, 'X'), (1, 'Y'), (2, 'X'))
        qasm = QiskitSimulation.get_qasm_header(3)
        qasm += QiskitSimulation.get_exponent_qasm(exp_operator_tuple, 1j * numpy.pi)
        qiskit_statevector = QiskitSimulation.get_statevector_from_qasm(qasm)

        # openfermion has reversed order of qubits?
        exp_operator_qo = 1j*openfermion.QubitOperator('X0 Y1 X2')
        exp_matrix = MatrixCalculation.get_qubit_operator_exponent_sparse_matrix(exp_operator_qo, 3, numpy.pi*2).todense()
        array_statevector = numpy.zeros(8)
        array_statevector[0] = 1
        array_statevector = exp_matrix.dot(array_statevector)

        self.assertEqual(len(array_statevector), len(qiskit_statevector))

        for i in range(len(qiskit_statevector)):
            self.assertEqual(qiskit_statevector[i], array_statevector[i])


if __name__ == '__main__':
    unittest.main()
