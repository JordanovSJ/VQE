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

        self.iter = None

        self.molecule_name = molecule.name

        self.n_electrons = molecule.n_electrons
        self.n_orbitals = molecule.n_orbitals
        self.n_qubits = self.n_orbitals

        self.molecule_data = MolecularData(geometry=molecule.geometry(** molecule_geometry_params),
                                           basis=basis, multiplicity=molecule.multiplicity, charge=molecule.charge)
        # logging.info('Running VQE for geometry {}'.format(self.molecule_data.geometry))

        self.molecule_psi4 = run_psi4(self.molecule_data, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=False)

        # Hamiltonian representations
        self.molecule_ham = self.molecule_psi4.get_molecular_hamiltonian()
        self.fermion_ham = get_fermion_operator(self.molecule_ham)
        self.jw_ham_qubit_operator = jordan_wigner(self.fermion_ham)
        self.jw_ham_sparse_matrix = get_sparse_operator(self.jw_ham_qubit_operator)  # need only this one

        self.energy = self.molecule_psi4.hf_energy.item()
        # logging.info('HF energy = {}'.format(self.energy))

        # TODO: this should be optional and used only when performing the matrix simulation. Put it somewhere else
        self.statevector = jw_hartree_fock_state(self.n_electrons, self.n_orbitals)

        if excitation_list is None:
            self.excitation_list = UCCSD(self.n_orbitals, self.n_electrons).get_excitation_list()
        else:
            # TODO this needs to be checked
            self.excitation_list = excitation_list

        self.var_params = numpy.zeros(len(self.excitation_list))

    # returns the compressed sparse row matrix for the exponent of a qubit operator
    def get_qubit_operator_exponent_matrix(self, qubit_operator, parameter=1, n_qubits=None):
        if n_qubits is None:
            n_qubits = self.n_qubits
        qubit_operator_matrix = get_sparse_operator(qubit_operator, n_qubits)
        return scipy.sparse.linalg.expm((parameter/2) * qubit_operator_matrix)  # TODO should we have 1j?

    @staticmethod
    def get_sparse_vector_module(sparse_vector):
        return numpy.sqrt(sparse_vector.conj().dot(sparse_vector.transpose()).todense().item())

    @staticmethod
    def renormalize_sparse_statevector(statevector):

        statevector_module = numpy.sqrt(statevector.conj().dot(statevector.transpose()).todense().item())
        assert statevector_module.imag == 0
        return statevector / statevector_module

    # update the ansatz statevector with new values of the var. params
    def update_statevector(self, params):  # TODO change name of this method

        assert len(self.excitation_list) == len(params)

        sparse_statevector = scipy.sparse.csr_matrix(jw_hartree_fock_state(self.n_electrons, self.n_orbitals))
        for i, excitation in enumerate(self.excitation_list):

            excitation_matrix = self.get_qubit_operator_exponent_matrix(excitation, parameter=params[i],
                                                                        n_qubits=self.n_qubits)
            sparse_statevector = sparse_statevector.dot(excitation_matrix.transpose())  # TODO: is transpose needed?

            # renormalize
            print('State vector module = ', self.get_sparse_vector_module(sparse_statevector))
            # sparse_statevector = self.renormalize_sparse_statevector(sparse_statevector)

        self.statevector = numpy.array(sparse_statevector.todense())[0]  # TODO: this is ugly -> fix

        return sparse_statevector

    # get the energy for new var. params
    def get_energy(self, excitation_params):
        sparse_statevector = self.update_statevector(excitation_params)
        bra = sparse_statevector.conj()
        ket = sparse_statevector.transpose()

        # ############### testing #############
        # print('bra')
        # print(bra.todense())
        # print('ham')
        # print(self.jw_ham_sparse_matrix.todense())
        # print('ket')
        # print(ket.todense())
        # #######################################

        energy = bra.dot(self.jw_ham_sparse_matrix).dot(ket)
        energy = energy.todense().item()
        print(energy)
        return energy.real  # TODO: should we expect the energy to be real ?

    def callback(self, xk):
        print('Iteration: {}'.format(self.iter))
        self.iter += 1
        print('Excitaiton parameters :', xk)

    def vqe_run(self, max_n_iterations=None):

        if max_n_iterations is None:
            max_n_iterations = len(self.excitation_list) * 100

        print('-----Running VQE for {}-----'.format(self.molecule_name))
        print('-----Number of electrons {}-----'.format(self.n_electrons))
        print('-----Number of orbitals {}-----'.format(self.n_orbitals))
        print('Qubit Hamiltonian: ', self.jw_ham_qubit_operator)

        parameters = numpy.zeros(len(self.excitation_list))

        # # ######## testing ##################
        # parameters[4] = -0.2233  # test for H2
        # print('####### testing ###########')
        # print(self.excitation_list[4])
        # self.get_energy(parameters)
        # print('###########################')
        # ##############################

        self.iter = 1
        opt_energy = scipy.optimize.minimize(self.get_energy, parameters, method='Nelder-Mead', callback=self.callback,
                                             options={'maxiter': max_n_iterations}, tol=1e-8)  # TODO: find a suitable optimizer

        return opt_energy
