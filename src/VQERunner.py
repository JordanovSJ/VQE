from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from openfermionpsi4 import run_psi4
from openfermion.hamiltonians import MolecularData
from openfermion.utils import jw_hartree_fock_state

import scipy
import numpy

import logging
from src.AnsatzType import UCCSD


class VQERunner:
    # Works for a single geometry
    def __init__(self, molecule, excitation_list=None, basis='sto-3g', molecule_geometry_params={}):

        # TODO! check attributes of MolecularData: n_electrons and n_orbitals
        self.molecule_data = MolecularData(geometry=molecule.geometry(** molecule_geometry_params),
                                           basis=basis, multiplicity=molecule.multiplicity, charge=molecule.charge )
        # logging.info('Running VQE for geometry {}'.format(self.molecule_data.geometry))

        # TODO: are molecular_ham and fermion_ham needed?
        self.molecule_psi4 = run_psi4(self.molecule_data, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=False)

        # Hamiltonian representations
        self.molecule_ham = self.molecule_psi4.get_molecular_hamiltonian()
        self.fermion_ham = get_fermion_operator(self.molecule_ham)
        self.jw_ham_qubit_operator = jordan_wigner(self.fermion_ham)
        self.jw_ham_sparse_matrix = get_sparse_operator(self.jw_ham_qubit_operator)  # need only this one

        # TODO initialise to HF ?
        self.energy = self.molecule_psi4.hf_energy.item()
        # logging.info('HF energy = {}'.format(self.energy))

        # TODO: this should be optional and used only when performing the matrix simulation. Put it somewhere else
        self.statevector = jw_hartree_fock_state(self.molecule_data.n_electrons, self.molecule_data.n_orbitals)

        # list of pairs: [QubitOperator, angle]
        if excitation_list is None:
            self.excitation_list = UCCSD(self.molecule_data.n_orbitals, self.molecule_data.n_electrons).\
                get_excitation_list()
        else:
            # TODO this needs to be checked
            self.excitation_list = excitation_list

    def update_energy(self):
        sparse_statevector = scipy.sparse.csr_matrix(self.statevector)
        self.energy = sparse_statevector.dot(self.jw_ham_sparse_matrix).dot(sparse_statevector.transpose())

    def get_energy(self, params):

        return 0

    def get_qubit_operator_matrix_exponential(self, qubit_operator, parameter=1, n_qubits=None):
        if n_qubits is None:
            n_qubits = self.molecule_data.n_qubits # TODO fixx!!!
        qubit_operator_matrix = get_sparse_operator(qubit_operator, n_qubits)
        return scipy.sparse.linalg.expm(-1j * parameter * qubit_operator_matrix)

    def get_excitation_matrix_list(self, excitation_list):
        excitation_matrix_list = []
        for excitation in excitation_list:
            excitation_matrix_list.append(self.get_qubit_operator_matrix_exponential(excitation))

        return excitation_matrix_list

    def vqe_run(self):

        parameters = numpy.zeros(len(self.excitation_list))
        opt_energy = scipy.optimize.minimize(self.get_energy, parameters)  # TODO arguments

        return opt_energy
