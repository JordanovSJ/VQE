from qiskit import QuantumCircuit, QuantumRegister
import random


class QubitOpUtils:

    # Takes in argument of a tuple of the form ((1, 'X'), (2, 'Z'), (3, 'X')) or ()
    # Returns a qasm_string that implements the gate
    @staticmethod
    def qasm_from_op_key(op_key: tuple, qreg_name='q'):
        if op_key == ():
            num_qubit = 0
        else:
            qubits = [gate[0] for gate in op_key]
            num_qubit = max(qubits) + 1 # Counts from 0

        qasm_str = ''
        qasm_str = QasmStrUtils.add_decl(qasm_str)
        qasm_str += '\nqreg {name}[{size}];'.format(name=qreg_name, size=num_qubit)

        for gate in op_key:
            qubit_index = gate[0]
            gate_type = gate[1].lower()
            assert gate_type == 'x' or 'y' or 'z'
            qasm_str += '\n{g_name} {q_name}[{q_index}];'.format(g_name=gate_type, q_name=qreg_name,
                                                                           q_index = qubit_index)

        return qasm_str

class QasmStrUtils:
    qasm_0 = 'OPENQASM 2.0;'
    qasm_def = '\ninclude "qelib1.inc";'

    # Composes qasm_str_1 and qasm_str_2 together in parallel
    # The two subcircuits occupy separate set of quantum registers
    # No classical registers
    # Returns composite qasm strings, and two list of new qreg names for the two sub-circuits
    @staticmethod
    def qasm_compose_parallel(qasm_str_1: str, qasm_str_2: str):
        # The function can't deal with classical registers in qasm_strings yet
        if 'creg' in qasm_str_1:
            raise Exception("qasm_str_1 contains classical register")
        elif 'creg' in qasm_str_2:
            raise Exception("qasm_str_2 contains classical register")

        qreg_names_1 = QasmStrUtils.qreg_names_from_qasm(qasm_str_1)
        qreg_names_2 = QasmStrUtils.qreg_names_from_qasm(qasm_str_2)
        # Add '[' to avoid confusion with other letters and substrings
        qreg_names_1 = [qreg_name + '[' for qreg_name in qreg_names_1]
        qreg_names_2 = [qreg_name + '[' for qreg_name in qreg_names_2]

        # The two qasm strings might contain the identical qreg names. This would cause
        # the two qasms to 'append' instead of 'compose'. It will also cause duplicate
        # declaration of quantum registers
        new_name_list_1 = []
        for qreg_name in qreg_names_1:
            new_name = 'q{}'.format(random.randint(0, 1000))
            while new_name in qreg_names_1:  # If the new name already is one of the qreg_names, regenerate a new name
                new_name = 'q{}'.format(random.randint(0, 1000))
            qasm_str_1 = qasm_str_1.replace(qreg_name, new_name + '[')
            new_name_list_1.append(new_name)

        new_name_list_2 = []
        for qreg_name in qreg_names_2:
            new_name = 'q{}'.format(random.randint(0, 1000))
            while new_name in qreg_names_2:
                new_name = 'q{}'.format(random.randint(0, 1000))
            qasm_str_2 = qasm_str_2.replace(qreg_name, new_name + '[')
            new_name_list_2.append(new_name)

        # Avoid duplicate declarations of qasm_0 and qasm_def
        qasm_str_2 = QasmStrUtils.remove_decl(qasm_str_2)

        # Compose the two strings
        qasm_str = qasm_str_1 + qasm_str_2

        return qasm_str, new_name_list_1, new_name_list_2

    # Appends qasm_str_2 to the right hand side of qasm_str_1
    # qasm_str_1 must have a qreg size at least as large as qasm_str_2
    @staticmethod
    def qasm_append(qasm_str_1: str, qasm_str_2: str):

        # The function can't deal with classical registers in qasm_strings yet
        if 'creg' in qasm_str_1:
            raise Exception("qasm_str_1 contains classical register")
        elif 'creg' in qasm_str_2:
            raise Exception("qasm_str_2 contains classical register")

        # Find the list of quantum register names of the two qasm strings
        qreg_names_1 = QasmStrUtils.qreg_names_from_qasm(qasm_str_1)
        if len(qreg_names_1) > 1:
            raise Exception("qasm_str_1 has multiple quantum registers")
        qreg_name_1 = qreg_names_1[0]

        # Find quantum register sizes of the two qasm_str
        qreg_size_1 = QasmStrUtils.qreg_size_from_qasm(qasm_str_1)
        qreg_size_2 = QasmStrUtils.qreg_size_from_qasm(qasm_str_2)

        if qreg_size_1 < qreg_size_2:
            raise Exception("qreg size of qasm_str_1 must be at least as large as qasm_str_2")

        if QasmStrUtils.qasm_def not in qasm_str_1:
            raise Exception("Gates not defined for qasm_str_1")

        # Remove qasm_0 and qasm_def from qasm_str_2, to avoid double declaration
        qasm_str_2 = QasmStrUtils.remove_decl(qasm_str_2)

        # Replace the qreg names
        qasm_str_2 = QasmStrUtils.rename_qasm(qasm_str_2, qreg_name_1)

        # Remove qreg declaration in qasm_str_2
        qasm_str_2 = QasmStrUtils.extract_qreg(qasm_str_2)[0]

        # Combine the two qasm strings
        qasm = qasm_str_1 + qasm_str_2

        return qasm

    # Takes in qasm_string, returns qasm_string for the inverted circuit
    @staticmethod
    def qasm_invert(qasm_str: str):
        qasm_str = QasmStrUtils.remove_decl(qasm_str)
        qasm_str, qasm_qreg = QasmStrUtils.extract_qreg(qasm_str)
        qasm = ''
        for line in reversed(qasm_str.splitlines()):
            qasm = qasm + '\n' + line
        qasm = qasm_qreg + qasm
        qasm_inverted = QasmStrUtils.add_decl(qasm)
        return qasm_inverted

    # Add measurement to all qubits in qasm_str
    @staticmethod
    def qasm_measure_all(qasm_str: str):
        # Generate new name for measure bits
        bit_name = 'measure{}'.format(random.randint(0, 1000))
        while bit_name in qasm_str:
            bit_name = 'measure{}'.format(random.randint(0, 1000))

        name_size_dict = QasmStrUtils.qreg_name_size_dicts_from_qasm(qasm_str)
        name_list = list(name_size_dict.keys())
        qreg_size = sum(name_size_dict.values())

        qasm_str = qasm_str + '\ncreg {name}[{size}];'.format(name=bit_name, size=qreg_size)
        measure_index = 0
        for name in name_list:
            for i in range(name_size_dict[name]):
                qasm_str = qasm_str + '\nmeasure {q_name}[{q_index}] -> {m_name}[{m_index}];'.format(
                    q_name=name, q_index=i, m_name=bit_name, m_index=measure_index
                )
                measure_index += 1

        return qasm_str

    # Construct qasm for swap test circuit
    @staticmethod
    def swap_test_qasm(qasm_psi, qasm_U):
        # Construct qasm for U U_psi |0>
        # This step will return error message if the qreg size don't match or if either of them
        # has multiple registers
        qasm_psi_U = QasmStrUtils.qasm_append(qasm_psi, qasm_U)

        # Construct qasm for ancilla qubit, which is control qubit for controlled-swap operation later
        qasm_ancilla = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg ancilla[1];\n'

        # Compose psi_U, psi, and ancilla together
        qasm_swap, new_name_ancilla_list, new_name_bulk_list = QasmStrUtils.qasm_compose_parallel(qasm_ancilla,
                                                                                                  QasmStrUtils.qasm_compose_parallel(
                                                                                                      qasm_psi_U,
                                                                                                      qasm_psi)[0])
        assert len(new_name_ancilla_list) == 1
        assert len(new_name_bulk_list) == 2
        new_name_ancilla = new_name_ancilla_list[0]
        new_name_bulk1 = new_name_bulk_list[0]
        new_name_bulk2 = new_name_bulk_list[1]

        name_size_dict = QasmStrUtils.qreg_name_size_dicts_from_qasm(qasm_swap)
        bulk_size = name_size_dict[new_name_bulk1]
        assert bulk_size == name_size_dict[new_name_bulk2]

        # Append h gate to ancilla qubit
        qasm_swap = qasm_swap + '\nh {}[0];'.format(new_name_ancilla)
        # Append cswap gates to the circuit
        for i in range(bulk_size):
            qasm_swap = qasm_swap + '\ncswap {ancilla_name}[0],' \
                                    '{q_name1}[{index1}],{q_name2}[{index2}];' \
                .format(ancilla_name=new_name_ancilla,
                        q_name1=new_name_bulk1, index1=i,
                        q_name2=new_name_bulk2, index2=i)
        # Append h gate to ancilla qubit
        qasm_swap = qasm_swap + '\nh {}[0];'.format(new_name_ancilla)

        # Add measurement to ancilla qubit
        qasm_swap = qasm_swap + '\ncreg measure0[1];\nmeasure {}[0] -> measure0[0];'.format(new_name_ancilla)

        return qasm_swap

    # Rename quantum register of a qasm_string
    @staticmethod
    def rename_qasm(qasm_str: str, new_qreg_name: str):
        old_names = QasmStrUtils.qreg_names_from_qasm(qasm_str)
        if len(old_names) != 1:
            raise Exception('qasm_str has multiple quantum registers')
        old_name = old_names[0] + '['
        new_name = new_qreg_name + '['

        qasm_str = qasm_str.replace(old_name, new_name)

        return qasm_str

    # Returns dictionary of name: size for the quantum register of the qasm string
    @staticmethod
    def qreg_name_size_dicts_from_qasm(qasm_str: str):
        qasm_qreg_list = QasmStrUtils.qreg_declare_list_from_qasm(qasm_str)
        qregs = {}
        for qasm_qreg in qasm_qreg_list:
            # qasm_qreg should look like 'qreg name[n];'
            index_1 = qasm_qreg.index('qreg')
            index_2 = qasm_qreg.index('[')
            index_3 = qasm_qreg.index(']')
            name = qasm_qreg[index_1 + 5: index_2]
            size = int(qasm_qreg[index_2 + 1: index_3])
            if name in qregs:
                raise Exception("Duplicate declaration of qreg under same name")
            qregs[name] = size

        return qregs

    # Return list of qreg names of a qasm string
    @staticmethod
    def qreg_names_from_qasm(qasm_str: str):
        qregs = QasmStrUtils.qreg_name_size_dicts_from_qasm(qasm_str)
        return list(qregs.keys())

    # Return quantum register total size of a qasm string
    @staticmethod
    def qreg_size_from_qasm(qasm_str: str):
        qregs = QasmStrUtils.qreg_name_size_dicts_from_qasm(qasm_str)
        return sum(qregs.values())

    # Returns list of qreg declaration lines from a qasm_str
    # Return type: list of str
    @staticmethod
    def qreg_declare_list_from_qasm(qasm_str: str):
        qasm_qreg = QasmStrUtils.extract_qreg(qasm_str)[1]
        qasm_qreg_list = qasm_qreg.splitlines()
        if '' in qasm_qreg_list:
            qasm_qreg_list.remove('')  # There may be empty lines in qasm_qreg
        for line in qasm_qreg_list:
            assert 'qreg' in line
        return qasm_qreg_list

    # Extract qreg definition line from qasm_str
    # Returns qasm_str without qreg definition lines, and a qasm_str with just the qreg definition lines
    @staticmethod
    def extract_qreg(qasm_str: str):
        qasm_qreg = ''
        while 'qreg' in qasm_str:
            # Partition qasm_str into a tuple of three strings with 'qreg'
            part_tuple_1 = qasm_str.partition('qreg')
            assert len(part_tuple_1) == 3

            # Partition the string on rhs of qreg with '];',
            # such that the first two strings of the second tuple are
            # the remaining half of the qreg declaration line
            part_tuple_2 = part_tuple_1[2].partition('];')
            qasm_str = part_tuple_1[0] + part_tuple_2[2]
            qasm_qreg += '\n' + part_tuple_1[1] + part_tuple_2[0] + part_tuple_2[1]

        return qasm_str, qasm_qreg

    # Remove qasm_0 and qasm_def from qasm_str
    @staticmethod
    def remove_decl(qasm_str: str):
        qasm_0 = QasmStrUtils.qasm_0
        qasm_def = QasmStrUtils.qasm_def
        if qasm_0 in qasm_str:
            qasm_str = qasm_str.replace(qasm_0, '')
        if qasm_def in qasm_str:
            qasm_str = qasm_str.replace(qasm_def, '')

        return qasm_str

    # Add qasm_0 and qasm_def to qasm_str
    @staticmethod
    def add_decl(qasm_str: str):
        qasm_0 = QasmStrUtils.qasm_0
        qasm_def = QasmStrUtils.qasm_def
        if qasm_def not in qasm_str:
            qasm_str = qasm_def + qasm_str
        if qasm_0 not in qasm_str:
            qasm_str = qasm_0 + qasm_str
        return qasm_str

    # Returns list of quantum register of a qasm string by constructing a circuit out of the qasm_str
    # Return type: list of QuantumRegister Objects
    @staticmethod
    def qreg_list_from_qasm_circ(qasm_str: str):
        # Create a circuit out of the qasm string
        circuit = QuantumCircuit.from_qasm_str(qasm_str)

        qreg_list = circuit.qregs

        return qreg_list
