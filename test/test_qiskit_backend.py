import unittest
import qiskit
import openfermion
from src.backends import QiskitSimulation
import numpy


class MyTestCase(unittest.TestCase):
    def test_pauli_gates_circuit_statevector(self):
        qubit_operator = openfermion.QubitOperator('X0 Y1')
        qasm_circuit = QiskitSimulation.get_qasm_header(2)
        qasm_circuit += QiskitSimulation.get_qubit_operator_qasm(qubit_operator)
        statevector = QiskitSimulation.get_statevector_from_qasm(qasm_circuit)

        expected_statevector = numpy.array([0, 0, 0, 1j])

        self.assertEqual(len(expected_statevector), len(statevector))

        for i in range(len(statevector)):
            self.assertEqual(statevector[i], expected_statevector[i])

    def test_get_energy(self):
        # TODO
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
