from openfermion.hamiltonians import MolecularData
from openfermion import get_fermion_operator, freeze_orbitals, jordan_wigner
from openfermionpsi4 import run_psi4

import numpy
import abc


class QSystem:

    def __init__(self, name, geometry, multiplicity, charge, n_orbitals, n_electrons, basis='sto-3g', frozen_els=None):
        self.name = name
        self.multiplicity = multiplicity
        self.charge = charge
        self.n_electrons = n_electrons
        self.basis = basis
        self.geometry = geometry

        self.molecule_data = MolecularData(geometry=self.geometry, basis=basis, multiplicity=self.multiplicity,
                                           charge=self.charge)

        self.molecule_psi4 = run_psi4(self.molecule_data, run_fci=True)

        # Hamiltonian transforms
        self.molecule_ham = self.molecule_psi4.get_molecular_hamiltonian()

        if frozen_els is None:
            self.n_orbitals = n_orbitals
            self.n_qubits = n_orbitals
            self.fermion_ham = get_fermion_operator(self.molecule_ham)
        else:
            self.n_orbitals = n_orbitals - len(frozen_els['occupied']) - len(frozen_els['unoccupied'])
            self.n_qubits = self.n_orbitals
            self.fermion_ham = freeze_orbitals(get_fermion_operator(self.molecule_ham), occupied=frozen_els['occupied'],
                                               unoccupied=frozen_els['unoccupied'], prune=True)
        self.jw_qubit_ham = jordan_wigner(self.fermion_ham)
        self.hf_energy = self.molecule_psi4.hf_energy.item()
        self.fci_energy = self.molecule_psi4.fci_energy.item()


class H2(QSystem):

    def __init__(self, r=0.735, basis='sto-3g', frozen_els=None):
        super(H2, self).__init__(name='H2', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=4,
                                 n_electrons=2, basis=basis, frozen_els=frozen_els)

    @staticmethod
    def get_geometry(r=0.735):
        return [['H', [0, 0, 0]],
                ['H', [0, 0, r]]]


class LiH(QSystem):

    def __init__(self, r=1.546, basis='sto-3g', frozen_els=None):
        super(LiH, self).__init__(name='LiH', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=12,
                                  n_electrons=4, basis=basis, frozen_els=frozen_els)

    @staticmethod
    def get_geometry(r=1.546):
        return [['Li', [0, 0, 0]],
                ['H', [0, 0, r]]]


class HF(QSystem):

    def __init__(self, r=0.995, basis='sto-3g', frozen_els=None):
        super(HF, self).__init__(name='HF', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=12,
                                 n_electrons=10, basis=basis, frozen_els=frozen_els)

    @staticmethod
    def get_geometry(r=0.995):
        return [['F', [0, 0, 0]],
                ['H', [0, 0, r]]]


class BeH2(QSystem):

    def __init__(self, r=1.316, basis='sto-3g', frozen_els=None):
        super(BeH2, self).__init__(name='BeH2', geometry=self.get_geometry(r), multiplicity=1, charge=0, n_orbitals=14,
                                   n_electrons=6, basis=basis, frozen_els=frozen_els)

    @staticmethod
    def get_geometry(r=1.316):
        return [['Be', [0, 0, 0]],
                ['H', [0, 0, r]],
                ['H', [0, 0, -r]]]


class H2O(QSystem):

    def __init__(self, r=1.316, theta=0.538*numpy.pi, basis='sto-3g', frozen_els=None):
        super(H2O, self).__init__(name='BeH2', geometry=self.get_geometry(r, theta), multiplicity=1, charge=0, n_orbitals=14,
                                  n_electrons=10, basis=basis, frozen_els=frozen_els)

    @staticmethod
    def get_geometry(r=1.0285, theta=0.538 * numpy.pi):
        return [
            ['O', [0, 0, 0]],
            ['H', [0, 0, -r]],
            ['H', [0, r*numpy.sin(numpy.pi - theta), r*numpy.cos(numpy.pi - theta)]]
        ]

#
# class H4(System):
#     name: str = 'H4'
#     multiplicity: int = 1
#     charge: int = 0
#     n_orbitals: int = 8
#     n_electrons: int = 4
#
#     @staticmethod
#     def geometry(distance=0.735):
#         return [
#             ['H', [0, 0, 0]],
#             ['H', [0, 0, distance]],
#             ['H', [0, 0, 2 * distance]],
#             ['H', [0, 0, 3 * distance]]
#           ]
