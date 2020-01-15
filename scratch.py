from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
from openfermion import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner, get_sparse_operator
from scipy.linalg import eigh
from openfermion.utils import jw_hartree_fock_state
import scipy
from src.AnsatzType import UCCSD
import numpy

if __name__ == "__main__":

    from src.Molecules import H2

    # choose basis for the molecular orbitals
    basis = 'sto-3g'

    molecule =H2

    h2_molecule = MolecularData(molecule.geometry(0.74), basis, molecule.multiplicity, molecule.charge)

    h2_molecule_psi4 = run_psi4(h2_molecule, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=False)

    molecular_ham = h2_molecule_psi4.get_molecular_hamiltonian()
    fermion_ham = get_fermion_operator(molecular_ham)
    jw_ham = jordan_wigner(fermion_ham)

    jw_ham_matrix = get_sparse_operator(jw_ham).todense()
    print(' JW Ham')
    print(jw_ham_matrix)
    eigenvalues, eigenvectors = eigh(jw_ham_matrix)

    assert len(jw_ham_matrix) == 2**h2_molecule_psi4.n_qubits

    statevector_0 = jw_hartree_fock_state(H2.n_electrons, H2.n_orbitals)
    print(' HF statevector ')
    print(statevector_0)

    sparse_statevector_0 = scipy.sparse.csr_matrix(statevector_0)

    excitation_list = UCCSD(H2.n_orbitals, H2.n_electrons).get_excitation_list()

    exc_n = 4

    print(' Exciation ', exc_n)
    print(excitation_list[exc_n])

    qubit_operator_matrix_excitation = get_sparse_operator(excitation_list[exc_n], H2.n_orbitals)

    parameter = 0.2215259843/2

    excitation_exponent_matrix_sparse = scipy.sparse.linalg.expm(-1j * parameter * qubit_operator_matrix_excitation)

    #### product of exponents rather than an exponent of sum TODO: maybe implement in the main code
    m = scipy.sparse.csr_matrix(numpy.identity(2**H2.n_orbitals))
    for item in excitation_list[exc_n].terms.items():
        qubit_operator_item = item[1]*QubitOperator(item[0])
        qubit_operator_item_matrix = get_sparse_operator(qubit_operator_item, H2.n_orbitals)
        m = scipy.sparse.linalg.expm_multiply(parameter*qubit_operator_item_matrix, m)  # no 1j
    ###

    excitation_exponent_matrix_sparse = m

    excitation_exponent_matrix = excitation_exponent_matrix_sparse.todense()
    print('Excitation matrix ')
    print(excitation_exponent_matrix)

    sparse_statevector = sparse_statevector_0.dot(excitation_exponent_matrix_sparse)

    print(' Excited statevector sparse')
    print(sparse_statevector.todense())

    statevector = excitation_exponent_matrix.dot(statevector_0)
    statevector_array = numpy.array(statevector)[0]
    statevector_array = statevector_array / numpy.sqrt(statevector_array.dot(statevector_array.conj()))
    print(' Excited statevector normalized')
    print(statevector_array)

    print('Energy ')
    print(statevector_array.conj().dot(jw_ham_matrix).dot(statevector.transpose()))

    print('Pizza')

