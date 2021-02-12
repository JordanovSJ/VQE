import numpy as np

# Import Qiskit
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit.quantum_info.operators import Operator


# Evaluates expectation value for a Hamiltonian given by Operator op_h
# Openfermion can get matrix forms of operators, and it is easy to generate operator from matrix
def ham_exp_value(qasm_psi, op_h: Operator, n_shots=1024, noisy=False,
                  noise_model=None, coupling_map=None):
    if noisy and noise_model is None:
        raise Exception("No noise model")

    # Construct circuit out of qasm_psi
    circ_psi = QuantumCircuit.from_qasm_str(qasm_str=qasm_psi)
    num_qubit_psi = len(circ_psi.qubits)

    # Construct circuit out of op_h
    if op_h.is_unitary():
        # if the H_matrix is a 2**m by 2**n (2**n columns) matrix, then
        # op_h.input_dims() gives a tuple of size n: (2,2,...2) with n 2s
        # and n is the number of qubits the operator acts on.
        if len(op_h.input_dims()) == num_qubit_psi:
            circ_U = op_to_circ(op_h)
        else:
            raise Exception("H_matrix and qasm_psi dimensions don't match")
    else:
        raise Exception("H is not unitary")

    exp_value = eval_exp_value_circ(circ_psi=circ_psi, circ_U=circ_U, n_shots=n_shots,
                                    noisy=noisy, noise_model=noise_model, coupling_map=coupling_map)
    return exp_value


# Evaluates expectation value for a Hamiltonian given by Operator op_h using swap test method
def ham_exp_value_swap(qasm_psi, op_h: Operator, n_shots=1024, noisy=False,
                       noise_model=None, coupling_map=None):
    if noisy and noise_model is None:
        raise Exception("No noise model")

    # Construct circuit out of qasm_psi
    circ_psi = QuantumCircuit.from_qasm_str(qasm_str=qasm_psi)
    num_qubit_psi = len(circ_psi.qubits)

    # Construct circuit out of op_h
    if op_h.is_unitary():
        # if the H_matrix is a 2**m by 2**n (2**n columns) matrix, then
        # op_h.input_dims() gives a tuple of size n: (2,2,...2) with n 2s
        # and n is the number of qubits the operator acts on.
        if len(op_h.input_dims()) == num_qubit_psi:
            circ_U = op_to_circ(op_h)
        else:
            raise Exception("H_matrix and qasm_psi dimensions don't match")
    else:
        raise Exception("H is not unitary")

    exp_value = eval_exp_value_swap_circ(circ_psi=circ_psi, circ_U=circ_U, n_shots=n_shots,
                                         noisy=noisy, noise_model=noise_model, coupling_map=coupling_map)
    return exp_value


# Converts an operator to a circuit
def op_to_circ(op_h):
    n_q = len(op_h.input_dims())
    qr_U = QuantumRegister(n_q)
    circ_U = QuantumCircuit(qr_U)
    circ_U.append(op_h, qr_U)
    return circ_U


# TO DO: What if initial state is not 00?
# Returns expectation value squared |<psi|U|psi>|^2=|<0|U_psi* U U_psi |0>|^2
def eval_exp_value(qasm_psi, qasm_U, n_shots=1024, noisy=False,
                   noise_model=None, coupling_map=None):
    if noisy and noise_model is None:
        raise Exception("No noise model")

    # Construct circuits for U_psi and U
    circ_psi = QuantumCircuit.from_qasm_str(qasm_str=qasm_psi)
    circ_U = QuantumCircuit.from_qasm_str(qasm_str=qasm_U)

    exp_value = eval_exp_value_circ(circ_psi=circ_psi, circ_U=circ_U, n_shots=n_shots,
                                    noisy=noisy, noise_model=noise_model,
                                    coupling_map=coupling_map)

    return exp_value


def eval_exp_value_circ(circ_psi, circ_U, n_shots=1024, noisy=False,
                        noise_model=None, coupling_map=None):
    if noisy and noise_model is None:
        raise Exception("No noise model")

    # Construct the circuit <psi|U|psi>
    circuit = circ_for_exp_value(circ_psi=circ_psi, circ_U=circ_U)

    if noisy:
        # Execute under noisy conditions
        counts = noisy_sim_circ(circ=circuit, noise_model=noise_model,
                                        coupling_map=coupling_map, n_shots=n_shots)
    else:
        # Execute in noiseless conditions
        counts = noiseless_sim_circ(circ=circuit, n_shots=n_shots)

    # THIS IS A VERY BAD WAY OF RETURNING THE RESULT. NEED TO THINK OF SOMETHING ELSE!
    n_qubits = len(circuit.qubits)
    string_all0 = "0" * n_qubits
    # This is constructing a string of all 0s for number of qubits,
    # in order to compare with key value in counts

    if string_all0 in counts:
        exp_value = counts[string_all0] / n_shots
    else:
        exp_value = 0

    return exp_value


# Construct circuit for U_psi U U_psi*
def circ_for_exp_value(circ_psi, circ_U):
    # Check register size
    n_q = len(circ_U.qubits)
    if n_q != len(circ_psi.qubits):
        raise Exception("Qubit register size of U and psi do not match")

    if len(circ_U.clbits) != 0 or len(circ_psi.clbits) != 0:
        raise Exception("Do not include classical register")

    # Construct circuit = U_psi U U_psi*
    circuit = circ_compose(circ_psi, circ_compose(circ_U, circ_psi.inverse()))
    qubits = circuit.qubits
    assert len(qubits) == n_q

    # Add classical bits to circuit to measure the qubits
    cr = ClassicalRegister(n_q)
    circuit.add_register(cr)
    circuit.measure(qubits, cr)

    return circuit


# Returns expectation value squared via SWAP test
# |<psi|U|psi>|^2
def eval_exp_value_swap(qasm_psi, qasm_U, n_shots=1024, noisy=False,
                        noise_model=None, coupling_map=None):
    if noisy and noise_model is None:
        raise Exception("No noise model")

    # Construct the circuit for |psi> and U|psi>
    circ_psi = QuantumCircuit.from_qasm_str(qasm_str=qasm_psi)
    circ_U = QuantumCircuit.from_qasm_str(qasm_str=qasm_U)

    exp_value = eval_exp_value_swap_circ(circ_psi=circ_psi, circ_U=circ_U, n_shots=n_shots,
                                         noisy=noisy, noise_model=noise_model,
                                         coupling_map=coupling_map)

    return exp_value


def eval_exp_value_swap_circ(circ_psi, circ_U, n_shots=1024, noisy=False,
                             noise_model=None, coupling_map=None):
    circuit = circ_swap_test(circ_psi=circ_psi, circ_U=circ_U)

    # Create qasm_string from the circuit
    # THIS DOESN'T WORK: the appended instructions appear in the qasm string as a undefined circuit
    # qasm_string = circuit.qasm()

    # Execute
    if noisy:
        counts = noisy_sim_circ(circuit, noise_model=noise_model,
                                        coupling_map=coupling_map, n_shots=n_shots)
    else:
        counts = noiseless_sim_circ(circuit, n_shots=n_shots)

    # P(first qubit = 0) = 1/2 + 1/2* |<psi|U|psi>|^2
    exp_value = 2 * (counts['0'] / n_shots - 1 / 2)

    return exp_value


# Constructs swap test circuit from qasm_psi and qasm_U
def circ_swap_test(circ_psi, circ_U):
    num_qubit_psi = len(circ_psi.qubits)
    num_qubit_U = len(circ_U.qubits)

    if num_qubit_U != num_qubit_psi:
        raise Exception("Qubit register size of U and psi do not match")

    circ_U_psi = circ_compose(circ_psi, circ_U)
    assert len(circ_U_psi.qubits) == num_qubit_U

    # Convert the circuits to instructions
    instr_U_psi = circ_U_psi.to_instruction()
    instr_psi = circ_psi.to_instruction()

    # For swap test we need one control qubit, and enough qubits for U|psi> and |psi>
    n_q = 1 + num_qubit_U + num_qubit_psi
    qr = QuantumRegister(n_q)
    cr = ClassicalRegister(1)  # Only one classical bit is needed

    # Create empty circuit to compose U_psi and psi together
    circuit = QuantumCircuit(qr, cr)

    # Append h gate to first qubit
    circuit.h(qr[0])

    # Append U_psi and psi to the circuit
    circuit.append(instr_U_psi, qr[1:num_qubit_U + 1])
    circuit.append(instr_psi, qr[num_qubit_U + 1:n_q])

    # Add swap gates to the circuit
    for i in range(num_qubit_U):
        circuit.cswap(qr[0], qr[1 + i], qr[1 + num_qubit_U + i])

    # Append second h gate to first qubit
    circuit.h(0)

    # Measure the control qubit on the only classical bit
    circuit.measure(qr[0], cr[0])

    return circuit


def noiseless_sim(qasm_str: str, n_shots=1024):
    # construct quantum circuit
    circ = QuantumCircuit.from_qasm_str(qasm_str=qasm_str)

    counts = noiseless_sim_circ(circ, n_shots=n_shots)

    return counts


# Same function as noiseless_sim, but with QuantumCircuit as argument.
# Python doesn't seem to support overloading.
def noiseless_sim_circ(circ: QuantumCircuit, n_shots=1024):
    # Select simulator to be Qasm from the Aer provider
    simulator = Aer.get_backend('qasm_simulator')

    # Execute under noiseless conditions and get counts
    result = execute(circ, simulator, shots=n_shots).result()
    # By default shots=1024. Can specify as well.
    counts = result.get_counts(circ)

    return counts


def noisy_sim(qasm_str: str, noise_model, coupling_map=None, n_shots=1024):
    # construct quantum circuit
    circ = QuantumCircuit.from_qasm_str(qasm_str=qasm_str)

    counts_noise = noisy_sim_circ(circ, noise_model=noise_model,
                                                coupling_map=coupling_map, n_shots=n_shots)

    return counts_noise


def noisy_sim_circ(circ: QuantumCircuit, noise_model, coupling_map=None, n_shots=1024):
    # Select simulator to be Qasm from the Aer provider
    simulator = Aer.get_backend('qasm_simulator')

    # TO ASK: WHAT ARE BASIS GATES?
    # Get basis gates for the noise model
    basis_gates = noise_model.basis_gates

    # Execute under noisy conditions
    result_noise = execute(circ, simulator, noise_model=noise_model,
                           basis_gates=basis_gates, coupling_map=coupling_map, shots=n_shots).result()
    counts_noise = result_noise.get_counts(circ)

    return counts_noise


# Returns circ1.compose(circ2), but by appending the circuits in form of instructions
def circ_compose(circ_1, circ_2):
    n_q = len(circ_1.qubits)
    if n_q != len(circ_2.qubits):
        raise Exception("Number of qubits of {} and {} do not match".format(circ_1, circ_2))

    # Construct a new circuit
    circuit = QuantumCircuit(n_q)
    qubits = circuit.qubits

    # Append U_psi, U and U_psi* to circuit as instructions,
    # compose method is unavailable
    instr_1 = circ_1.to_instruction()
    instr_2 = circ_2.to_instruction()

    circuit.append(instr_1, qubits)
    circuit.append(instr_2, qubits)

    return circuit
