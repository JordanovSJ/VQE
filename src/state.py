
# recipe to initialize some state
class State:
    def __init__(self, elements, parameters, n_qubits, n_electrons, init_state_qasm=None):
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.elements = elements
        self.parameters = parameters
        self.init_state_qasm = init_state_qasm

    def add_element(self, element, parameter):
        self.elements.append(element)
        self.parameters.apppend(parameter)
