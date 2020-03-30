import unittest

from src.utils import QasmUtils
from src.backends import QiskitSimulation

import numpy


class QasmUtilsTest(unittest.TestCase):

    def test_controlled_y_gate_1(self):
        control = 0
        target = 1

        # case 1
        qasm = QasmUtils.qasm_header(2)
        qasm += 'x q[{}];\n'.format(control)
        qasm += QasmUtils.controlled_y_gate(numpy.pi / 2, control, target)
        statevector = QiskitSimulation.get_statevector_from_qasm(qasm).round(3)

        expected_statevector = numpy.array([0, 1, 0, 1])/numpy.sqrt(2)
        expected_statevector = expected_statevector.round(3)

        for i in range(len(statevector)):
            self.assertEqual(statevector[i], expected_statevector[i])

    def test_controlled_y_gate_2(self):
        control = 0
        target = 1

        #case 2
        qasm = QasmUtils.qasm_header(2)
        qasm += 'x q[{}];\n'.format(control)
        qasm += 'x q[{}];\n'.format(target)
        qasm += QasmUtils.controlled_y_gate(numpy.pi / 2, control, target)
        statevector = QiskitSimulation.get_statevector_from_qasm(qasm).round(3)

        expected_statevector = numpy.array([0, -1, 0, 1]) / numpy.sqrt(2)
        expected_statevector = expected_statevector.round(3)

        for i in range(len(statevector)):
            self.assertEqual(statevector[i], expected_statevector[i])

    def test_partial_exchange(self):
        qubit_1 = 0
        qubit_2 = 1

        qasm_1 = QasmUtils.qasm_header(2)
        qasm_1 += 'x q[{}];\n'.format(qubit_1)

        qasm_2 = QasmUtils.qasm_header(2)
        qasm_2 += 'x q[{}];\n'.format(qubit_2)

        angles = [0, numpy.pi/4, numpy.pi/3, numpy.pi/2, numpy.pi]

        for angle in angles:

            statevector_1 = QiskitSimulation.\
                get_statevector_from_qasm(qasm_1 + QasmUtils.partial_exchange_gate(angle, qubit_1, qubit_2)).round(3)

            expected_statevector_1 = numpy.array([0, numpy.cos(angle), numpy.sin(angle), 0])
            expected_statevector_1 = expected_statevector_1.round(3)

            for i in range(len(statevector_1)):
                self.assertEqual(statevector_1[i], expected_statevector_1[i])

            statevector_2 = QiskitSimulation.\
                get_statevector_from_qasm(qasm_2 + QasmUtils.partial_exchange_gate(angle, qubit_1, qubit_2)).round(3)

            expected_statevector_2 = numpy.array([0, - numpy.sin(angle), numpy.cos(angle), 0])
            expected_statevector_2 = expected_statevector_2.round(3)

            for i in range(len(statevector_2)):
                self.assertEqual(statevector_2[i], expected_statevector_2[i])

    def test_parity_controlled_phase_gate_1(self):

        qasm = ['']
        qasm.append(QasmUtils.qasm_header(3))
        qasm.append('h q[0];\n x q[1];\n x q[2];\n')
        qasm.append(QasmUtils.parity_controlled_phase_gate([0], 1, 2))
        statevector = QiskitSimulation.get_statevector_from_qasm(''.join(qasm))

        self.assertEqual(statevector[-1].round(5).real, (1/numpy.sqrt(2)).round(5))
        self.assertEqual(statevector[-2].round(5).real, (-1/numpy.sqrt(2)).round(5))

        # test the reverse control case
        qasm = ['']
        qasm.append(QasmUtils.qasm_header(3))
        qasm.append('h q[0];\n x q[1];\n x q[2];\n')
        qasm.append(QasmUtils.parity_controlled_phase_gate([0], 1, 2, reverse=True))
        statevector = QiskitSimulation.get_statevector_from_qasm(''.join(qasm))

        self.assertEqual(statevector[-1].round(5).real, (-1 / numpy.sqrt(2)).round(5))
        self.assertEqual(statevector[-2].round(5).real, (1 / numpy.sqrt(2)).round(5))

    def test_parity_controlled_phase_gate_2(self):
        qasm = ['']
        qasm.append(QasmUtils.qasm_header(4))
        qasm.append('h q[0];\n h q[1];\n x q[2];\n x q[3];\n')
        qasm.append(QasmUtils.parity_controlled_phase_gate([0, 1], 2, 3))
        statevector = QiskitSimulation.get_statevector_from_qasm(''.join(qasm))

        self.assertEqual(statevector[-1].real.round(5), (-1 / 2))  # -1111
        self.assertEqual(statevector[-2].real.round(5), (1 / 2))  # 1110
        self.assertEqual(statevector[-3].real.round(5), (1 / 2))  # 1101
        self.assertEqual(statevector[-4].real.round(5), (-1 / 2))  # -1100

        # test the reverse control case
        qasm = ['']
        qasm.append(QasmUtils.qasm_header(4))
        qasm.append('h q[0];\n h q[1];\n x q[2];\n x q[3];\n')
        qasm.append(QasmUtils.parity_controlled_phase_gate([0, 1], 2, 3, reverse=True))
        statevector = QiskitSimulation.get_statevector_from_qasm(''.join(qasm))

        self.assertEqual(statevector[-1].real.round(5), (1 / 2))
        self.assertEqual(statevector[-2].real.round(5), (-1 / 2))
        self.assertEqual(statevector[-3].real.round(5), (-1 / 2))
        self.assertEqual(statevector[-4].real.round(5), (1 / 2))


if __name__ == '__main__':
    unittest.main()
