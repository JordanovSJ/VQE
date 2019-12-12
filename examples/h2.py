# class for a H2 molecule


class Molecule:
    name: str = 'H2'
    multiplicity: int = 1
    charge: int = 0
    n_orbitals: int = 4
    n_electrons: int = 2

    @staticmethod
    def geometry(distance=1):
        return [
                ['H', [0, 0, 0]],
                ['H', [0, 0, distance]]]


molecule = Molecule()

