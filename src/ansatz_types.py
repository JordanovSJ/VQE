from openfermion import QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner
import itertools

# TODO create a parent class for the different ansatz type classes
# class Ansatz:
#     def __init__(self):
#         return 0
#


class UCCSD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

    def get_single_excitation_list(self):
        single_excitations = []
        for i in range(self.n_electrons):
            for j in range(self.n_electrons, self.n_orbitals):
                fermion_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(j, i))
                single_excitations.append(jordan_wigner(fermion_operator))

        # returns list of QubitOperators
        return single_excitations

    def get_double_excitation_list(self):
        double_excitations = []
        for i in range(self.n_electrons-1):
            for j in range(i+1, self.n_electrons):
                for k in range(self.n_electrons, self.n_orbitals-1):
                    for l in range(k+1, self.n_orbitals):
                        fermion_operator = FermionOperator('[{2}^ {3}^ {0} {1}] - [{0}^ {1}^ {2} {3}]'
                                                           .format(i, j, k, l))
                        double_excitations.append(jordan_wigner(fermion_operator))

        return double_excitations

    def get_excitation_list(self):
        return self.get_single_excitation_list() + self.get_double_excitation_list()


class UCCGSD:
    def __init__(self, n_orbitals, n_electrons):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons

    def get_single_excitation_list(self):
        single_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 2):
            fermion_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(* indices))
            single_excitations.append(jordan_wigner(fermion_operator))

        return single_excitations

    def get_double_excitation_list(self):
        double_excitations = []
        for indices in itertools.combinations(range(self.n_orbitals), 4):
            fermion_operator = FermionOperator('[{1}^ {0}] - [{0}^ {1}]'.format(* indices))
            double_excitations.append(jordan_wigner(fermion_operator))

        return double_excitations

    def get_excitation_list(self):
        return self.get_single_excitation_list() + self.get_double_excitation_list()

# class k-UpCCGSD:
#     #TODO
