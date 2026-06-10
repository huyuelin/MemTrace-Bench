from enum import Enum


class ValidityLattice(Enum):
    """Validity lattice: Drop < Obligation < Hypothesis < Fact"""

    DROP = 0
    OBLIGATION = 1
    HYPOTHESIS = 2
    FACT = 3

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value


def join(a: ValidityLattice, b: ValidityLattice) -> ValidityLattice:
    """Lattice join (least upper bound)."""
    return a if a.value >= b.value else b


def meet(a: ValidityLattice, b: ValidityLattice) -> ValidityLattice:
    """Lattice meet (greatest lower bound)."""
    return a if a.value <= b.value else b
