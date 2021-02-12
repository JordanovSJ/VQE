from qiskit import IBMQ
from qiskit.providers.aer.noise import device, NoiseModel
import qiskit.providers.aer.noise.errors as errors
from qiskit.providers.aer.noise.errors import pauli_error, reset_error, thermal_relaxation_error
import numpy as np

class TestUtils:

    # Returns noise model and coupling map of ibmq_vigo device
    @staticmethod
    def vigo_noise_model():
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')

        backend = provider.get_backend('ibmq_vigo')

        noise_model = device.basic_device_noise_model(backend.properties())
        coupling_map = backend.configuration().coupling_map

        return noise_model, coupling_map

    # prob_1 and prob_2 are error prob for single and two qubit gates
    # prob_bit_flip and prob_phase_flip are bit_flip and phase_flip probabilities for Pauli error
    # p_meas is measurement error, for a prob of p_meas the measurement of 1 gives 0 and 0 gives 1
    # These error are added to all single, double qubit gates and measure gates
    @staticmethod
    def custom_noise_model(prob_1=0.001, prob_2=0.01, prob_bit_flip=0.05, prob_phase_flip=0.05,
                           p_meas=0.1):
        # Create empty noise model
        noise_model_custom = NoiseModel()

        # Depolorizing errors for single and two qubit gates
        error_1 = errors.depolarizing_error(prob_1, num_qubits=1)  # For single qubit gates
        error_2 = errors.depolarizing_error(prob_2, num_qubits=2)  # For two qubit gates

        # Pauli error
        bit_flip = pauli_error([('X', prob_bit_flip), ('I', 1 - prob_bit_flip)])  # Bit flip
        phase_flip = pauli_error([('Z', prob_phase_flip), ('I', 1 - prob_phase_flip)])  # Phase flip
        bitphase_flip = bit_flip.compose(phase_flip)

        # Compose the errors
        error_gate1 = bitphase_flip.compose(error_1)  # For single qubit gates
        error_gate2 = bitphase_flip.tensor(bitphase_flip).compose(error_2)  # For two qubit gates

        # Measurement error: for a prob of p_meas the measurement of 1 gives 0 and 0 gives 1,
        # i.e. a bit flip on measurement gate
        error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])

        # Add errors to the noise model
        noise_model_custom.add_all_qubit_quantum_error(error_gate1, ['u1', 'u2', 'u3'])
        noise_model_custom.add_all_qubit_quantum_error(error_gate2, ['cx'])
        noise_model_custom.add_all_qubit_quantum_error(error_meas, ['measure'])

        return noise_model_custom

    # This method creates a thermal noise model for n_q qubits for testing purposes.
    # For future benchmarking this should be made into a proper function
    @staticmethod
    def thermal_noise_model(n_q):
        # Create empty noise model
        noise_thermal = NoiseModel()

        # T1 and T2 values for qubits 0-3
        T1s = np.random.normal(50e3, 10e3, n_q)  # Sampled from normal distribution mean 50 microsec
        T2s = np.random.normal(50e3, 10e3, n_q)  # Sampled from normal distribution mean 50 microsec

        # Truncate random T2s <= T1s
        T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(n_q)])

        # Instruction times (in nanoseconds)
        time_u1 = 0  # virtual gate
        time_u2 = 50  # (single X90 pulse)
        time_u3 = 100  # (two X90 pulses)
        time_cx = 300
        time_measure = 1000  # 1 microsecond

        # QuantumError objects
        errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                          for t1, t2 in zip(T1s, T2s)]
        errors_u1 = [thermal_relaxation_error(t1, t2, time_u1)
                     for t1, t2 in zip(T1s, T2s)]
        errors_u2 = [thermal_relaxation_error(t1, t2, time_u2)
                     for t1, t2 in zip(T1s, T2s)]
        errors_u3 = [thermal_relaxation_error(t1, t2, time_u3)
                     for t1, t2 in zip(T1s, T2s)]
        errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
            thermal_relaxation_error(t1b, t2b, time_cx))
            for t1a, t2a in zip(T1s, T2s)]
            for t1b, t2b in zip(T1s, T2s)]

        for j in range(n_q):
            noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
            noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
            noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
            noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
            for k in range(n_q):
                noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

        return noise_thermal
