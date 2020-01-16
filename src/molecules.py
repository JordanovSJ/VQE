from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4

import abc

# TODO make example molecules child classes?
# class Molecule:
#     name: str
#     multiplicity: int
#     charge: int
#     n_orbitals: int
#     n_electrons: int
#
#     @staticmethod
#     @abc.abstractmethod
#     def geometry():
#         pass
#
#     def molecular_data(self, basis):
#         return MolecularData((self.geometry(), basis, self.multiplicity, self.charge))
#
#     def molecule_psi4(self, basis):
#         return run_psi4(self.molecular_data(basis))


class H2:
    name: str = 'H2'
    multiplicity: int = 1
    charge: int = 0
    n_orbitals: int = 4
    n_electrons: int = 2
    # ground_state_distance = 0.735

    @staticmethod
    def geometry(distance=1):

        return [
                ['H', [0, 0, 0]],
                ['H', [0, 0, distance]]]


class LiH:
    name: str = 'LiH'
    multiplicity: int = 1
    charge: int = 0
    n_orbitals: int = 12
    n_electrons: int = 4

    # ground_state_distance = 1.547

    @staticmethod
    def geometry(distance=1):
        return [
            ['Li', [0, 0, 0]],
            ['H', [0, 0, distance]]]


class HF:
    name: str = 'HF'
    multiplicity: int = 1
    charge: int = 0
    n_orbitals: int = 12
    n_electrons: int = 10
    # ground_state_distance = 0.995

    @staticmethod
    def geometry(distance=1):
        return [
            ['F', [0, 0, 0]],
            ['H', [0, 0, distance]]]

