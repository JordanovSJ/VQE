import openfermion
from openfermion import FermionOperator as FO
from src.molecular_system import MolecularSystem


def ham_16_qubits():
    Vbc = [[.0250807169923074, -.0868144072950623, .0085583025756312, -.0453252197425094, -.0994068127537739,
            -.0806228772741335],
           [-.0457200481761811, .0099115160477145, .0139115378406957, -.0861511722789291, -.0260372216384797,
            -.0508725514072142]]

    Eb = [[.1614803305373006, .5965506376046160, .4886895836185136, .3258340605965911, .3602436300555840,
           .4402214440260793],
          [.5965506376046160, 1.1229456340538546, .8529290249653549, .7141792453632360, .5956125332489218,
           .8892400572481333],
          [.4886895836185136, .8529290249653549, .5162948080439264, .5684505485341950, .4841907805435753,
           .7141818571126369],
          [.3258340605965911, .7141792453632360, .5684505485341950, .3375764688659696, .3043939386462531,
           .4806909888771593],
          [.3602436300555840, .5956125332489218, .4841907805435753, .3043939386462531, .0937845220196198,
           .3788994588642509],
          [.4402214440260793, .8892400572481333, .7141818571126369, .4806909888771593, .3788994588642509,
           .6362487049018987]]

    H = 0.294 * FO('[0^ 0 1^ 1]')
    H += 0.294 * FO('[2^ 2 3^ 3]')
    H += -0.6009694 * (FO('[0^ 0]') + FO('[1^ 1]'))
    H += -0.5800571 * (FO('[2^ 2]') + FO('[3^ 3]'))

    for f in range(2):
        for d in range(6):
            factor = Vbc[f][d]
            # TODO check
            term_even = FO('[{0}^ {1}] - [{1}^ {0}]'.format(2 * f, 2 * (2 + d)))
            term_odd = FO('[{0}^ {1}] - [{1}^ {0}]'.format(2 * f + 1, 2 * (2 + d) + 1))

            H += factor * (term_odd + term_even)

    for d1 in range(6):
        for d2 in range(6):
            factor = Eb[d1][d2]
            # TODO check
            term_even = FO('[{0}^ {1}] - [{1}^ {0}]'.format(2 * (d1 + 2), 2 * (d2 + 2)))
            term_odd = FO('[{0}^ {1}] - [{1}^ {0}]'.format(2 * (d1 + 2) + 1, 2 * (d2 + 2) + 1))

            H += factor * (term_odd + term_even)

    return H


def ham_14_qubits():

    Vbc = [[-.0204661937105278, -.0117290619116000, -.0465447584618736, .0570225547542926, .0673621582334465],
           [-.0279153487228577, .0408511261173111, -.0685495682013786, -.0388639034459344, -.0103339223672084]]

    Eb = [[-.1035627634782331, .0160071527143417, -.0945067764959269, .0139349034371563, .0073931708614382],
          [.0160071527143417, -.0702933662507285, .0087701950185232, -.0297422158609423, -.0472917374161863],
          [ -.0945067764959269, .0087701950185232, -.1529062013247139, -.0207580947707079, .0077200188411109],
          [.0139349034371563, -.0297422158609423, -.0207580947707079, -.0945758726824176, -.0944388023691522],
          [.0073931708614382, -.0472917374161863, .0077200188411109, -.0944388023691522, -.0982903717643146]]
#
    U = 0.2940000
    eps_1 = -0.284183
    eps_2 = -0.2632707
    n_bath = 5
    n_impurity = 2

    H = U * FO('[0^ 0 1^ 1]')
    H += U * FO('[2^ 2 3^ 3]')
    H += eps_1 * (FO('[0^ 0]') + FO('[1^ 1]'))
    H += eps_2 * (FO('[2^ 2]') + FO('[3^ 3]'))

    for f in range(n_impurity):
        for d in range(n_bath):
            factor = Vbc[f][d]
            # spin up
            term_even = FO('[{0}^ {1}] + [{1}^ {0}]'.format(2 * f, 2 * (n_impurity + d)))
            # spin down
            term_odd = FO('[{0}^ {1}] + [{1}^ {0}]'.format(2 * f + 1, 2 * (n_impurity + d) + 1))

            H += factor * (term_odd + term_even)

    for d1 in range(n_bath):
        for d2 in range(n_bath):
            factor = Eb[d1][d2]

            # if d1 == d2:
            # spin up
            term_even = FO('[{0}^ {1}]'.format(2 * (d1 + n_impurity), 2 * (d2 + n_impurity)))
            # spin down
            term_odd = FO('[{0}^ {1}]'.format(2 * (d1 + n_impurity) + 1, 2 * (d2 + n_impurity) + 1))
            # else:
            #
            #     # spin up
            #     term_even = 0.5*FO('[{0}^ {1}] + [{1}^ {0}]'.format(2 * (d1 + n_impurity), 2 * (d2 + n_impurity)))
            #     # spin down
            #     term_odd = 0.5*FO('[{0}^ {1}] + [{1}^ {0}]'.format(2 * (d1 + n_impurity) + 1, 2 * (d2 + n_impurity) + 1))

            H += factor * (term_odd + term_even)

    print(len(H.terms))

    return H


class ElectronicSystem:
    def __init__(self, fermi_ham, n_orbitals, n_electrons):
        self.name = 'Pizza'
        self.n_electrons = n_electrons

        self.n_orbitals = n_orbitals
        self.n_qubits = self.n_orbitals
        self.fermi_ham = fermi_ham
        self.qubit_ham = openfermion.jordan_wigner(self.fermi_ham)

        self.hf_energy = 0  # wild guess

        self.H_lower_state_terms = None


