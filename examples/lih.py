# class for a H2 molecule


class Molecule:
    name: str = 'LiH'
    multiplicity: int = 1
    charge: int = 0
    n_orbitals: int = 12
    n_electrons: int = 4

    @staticmethod
    def geometry(distance=1):
        return [
                ['Li', [0, 0, 0]],
                ['H', [0, 0, distance]]]


molecule = Molecule()

