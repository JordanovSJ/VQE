import numpy
import pandas

from src.iter_vqe_utils import DataUtils
from src.ansatz_elements import SQExc, DQExc
from src.state import State
from src.q_system import QSystem


class H2(QSystem):

    def __init__(self, r=0.735, basis='sto-3g', frozen_els=None):
        super(H2, self).__init__(name='H2', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=4,
                                 n_electrons=2, basis=basis, frozen_els=frozen_els)

    # the ground and the first three degenerate excited states for H2 in equilibrium configuration
    # this is used for excited state simulations only
    def default_states(self):
        ground = State([DQExc([0, 1], [2, 3])], [0.11176849919227788], 4, 2)
        first_exc_1 = State([SQExc(0, 3)], [1.570796325683595], 4, 2)
        first_exc_2 = State([SQExc(1, 2)], [1.570796325683595], 4, 2)
        first_exc_3 = State([DQExc([0, 1], [2, 3]), SQExc(1, 3)], [-numpy.pi / 4, numpy.pi / 2], 4, 2)
        self.H_lower_state_terms = [[factor, state] for factor, state in
                                    zip([2.2, 1.55, 1.55, 1.55], [ground, first_exc_1, first_exc_2, first_exc_3])]
        return self.H_lower_state_terms

    @staticmethod
    def get_geometry(r=0.735):
        return [['H', [0, 0, 0]],
                ['H', [0, 0, r]]]


class H4(QSystem):

    def __init__(self, r=0.735, basis='sto-3g', frozen_els=None):
        super(H4, self).__init__(name='H4', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=8,
                                 n_electrons=4, basis=basis, frozen_els=frozen_els)

    @staticmethod
    def get_geometry(distance=0.735):
        return [
            ['H', [0, 0, 0]],
            ['H', [0, 0, distance]],
            ['H', [0, 0, 2 * distance]],
            ['H', [0, 0, 3 * distance]]
          ]


class LiH(QSystem):
    # frozen_els = {'occupied': [0,1], 'unoccupied': []}
    def __init__(self, r=1.546, basis='sto-3g', frozen_els=None):
        super(LiH, self).__init__(name='LiH', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=12,
                                  n_electrons=4, basis=basis, frozen_els=frozen_els)

    def default_states(self):
        df = pandas.read_csv('src/molecules/LiH_h_adapt_gsdqe_comp_pairs_15-Sep-2020.csv')
        ground = DataUtils.ansatz_from_data_frame(df, self)
        del df
        self.H_lower_state_terms = [[abs(self.hf_energy)*2, ground]]
        return self.H_lower_state_terms

    @staticmethod
    def get_geometry(r=1.546):
        return [['Li', [0, 0, 0]],
                ['H', [0, 0, r]]]


class HF(QSystem):

    def __init__(self, r=0.995, basis='sto-3g', frozen_els=None,):
        super(HF, self).__init__(name='HF', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=12,
                                 n_electrons=10, basis=basis, frozen_els=frozen_els,)

    @staticmethod
    def get_geometry(r=0.995):
        return [['F', [0, 0, 0]],
                ['H', [0, 0, r]]]


class BeH2(QSystem):
    # frozen_els = {'occupied': [0,1], 'unoccupied': [6,7]}
    def __init__(self, r=1.316, basis='sto-3g', frozen_els=None,):
        super(BeH2, self).__init__(name='BeH2', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=14,
                                   n_electrons=6, basis=basis, frozen_els=frozen_els,)

    def default_states(self):
        df = pandas.read_csv('../../src/molecules/BeH2_h_adapt_gsdqe_comp_pairs_15-Sep-2020.csv')
        ground = DataUtils.ansatz_from_data_frame(df, self)
        del df
        self.H_lower_state_terms = [[abs(self.hf_energy)*2, ground]]

    @staticmethod
    def get_geometry(r=1.316):
        return [['Be', [0, 0, 0]],
                ['H', [0, 0, r]],
                ['H', [0, 0, -r]]]


class H2O(QSystem):

    def __init__(self, r=1.0285, theta=0.538*numpy.pi, basis='sto-3g', frozen_els=None):
        super(H2O, self).__init__(name='H20', geometry=self.get_geometry(r, theta), multiplicity=1, charge=0, n_orbitals=14,
                                  n_electrons=10, basis=basis, frozen_els=frozen_els,)

    @staticmethod
    def get_geometry(r=1.0285, theta=0.538 * numpy.pi):
        return [
            ['O', [0, 0, 0]],
            ['H', [0, 0, -r]],
            ['H', [0, r*numpy.sin(numpy.pi - theta), r*numpy.cos(numpy.pi - theta)]]
        ]


