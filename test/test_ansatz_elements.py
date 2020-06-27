import unittest

from src.utils import QasmUtils
from src.backends import QiskitSim
from src.ansatz_elements import DoubleExchange

import numpy


class AnsatzElementsTest(unittest.TestCase):

    def test_double_exchange_extended(self):
        angles = [- 0.2, 0.2]

        for angle in angles:

            qasm = ['']
            qasm.append(QasmUtils.qasm_header(6))
            # qasm_1.append('x q[3];\n')
            qasm.append('h q[4];\n')
            qasm.append('x q[2];\n')
            qasm.append('x q[0];\n')

            qasm.append(DoubleExchange([0, 2], [3, 5], rescaled_parameter=True, parity_dependence=True).get_qasm([angle]))
            statevector = QiskitSim.statevector_from_qasm(''.join(qasm))

            self.assertEqual(statevector[56].real.round(5), - statevector[40].real.round(5))

            qasm = ['']
            qasm.append(QasmUtils.qasm_header(6))
            # qasm_1.append('x q[3];\n')
            qasm.append('h q[4];\n')
            qasm.append('x q[2];\n')
            qasm.append('x q[0];\n')

            qasm.append(DoubleExchange([0, 2], [3, 5], rescaled_parameter=True).get_qasm([angle]))
            statevector = QiskitSim.statevector_from_qasm(''.join(qasm))

            self.assertEqual(statevector[56].real.round(5), statevector[40].real.round(5))
