"""Models for the sequence of repeat units along a polymer chain"""

from typing import Mapping, Sequence
from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

from .._common import UnitLabel


class SequenceModel(ABC):
    """Abstract base class for drawing sequences of repeat-unit labels"""

    @property
    @abstractmethod
    def labels(self) -> frozenset[UnitLabel]:
        """Every unit label this model can emit"""
        ...

    @abstractmethod
    def sample(self, n_units: int, rng: np.random.Generator) -> list[UnitLabel]:
        """Draw a sequence of exactly n_units unit labels"""
        ...


def _normalized(weights: Mapping[UnitLabel, float]) -> dict[UnitLabel, float]:
    """Normalize non-negative weights to probabilities"""
    total = sum(weights.values())
    if any(weight < 0 for weight in weights.values()) or total <= 0:
        raise ValueError(f"Weights must be non-negative with a positive sum, got {dict(weights)}")
    return {label: weight / total for label, weight in weights.items()}

class SequenceHomopolymer(SequenceModel):
    """Homopolymer sequence"""
    def __init__(self, unit_label: UnitLabel) -> None:
        self.unit_label = unit_label

    @property
    def labels(self) -> frozenset[UnitLabel]:
        return frozenset([self.unit_label])

    def sample(self, n_units: int, rng: Generator) -> list[UnitLabel]:
        return [self.unit_label for _ in range(n_units)]

    def __repr__(self) -> str:
        return f"SequenceHomopolymer({self.unit_label})"

class SequenceRandom(SequenceModel):
    """Each unit drawn independently in proportion to the given ratios (Bernoulli statistics)"""

    def __init__(self, ratios: Mapping[UnitLabel, float]) -> None:
        self.probabilities = _normalized(ratios)

    @property
    def labels(self) -> frozenset[UnitLabel]:
        return frozenset(self.probabilities)

    def sample(self, n_units: int, rng: np.random.Generator) -> list[UnitLabel]:
        labels = list(self.probabilities)
        # draw indices (not labels) so labels come back as native Python objects
        idxs = rng.choice(len(labels), size=n_units, p=list(self.probabilities.values()))
        return [labels[i] for i in idxs]

    def __repr__(self) -> str:
        return f"SequenceRandom({self.probabilities})"

class SequenceMayoLewis(SequenceModel):
    """
    Terminal (Mayo-Lewis) copolymerization model, generalized to any number of units

    In the terminal model, the rate at which a chain adds its next unit depends only on
    the unit at its growing end. The sequence is then a Markov chain: unit i is followed
    by unit j with a probability set by the feed and the reactivity ratios.

    Parameters
    ----------
    feed : Mapping[UnitLabel, float]
        Relative amounts of each unit in the monomer feed (normalized, so {'S': 1, 'M': 1}
        and {'S': 0.5, 'M': 0.5} are the same)
    reactivity : Mapping[(UnitLabel, UnitLabel), float], optional
        Reactivity RATIOS (not rate constants), keyed by ordered pair:

            r_ij = k_ii / k_ij

        how much faster a chain ending in i adds another i than it adds a j. r_ij > 1
        favors runs of i; r_ij < 1 favors i -> j steps. Missing pairs default to 1 (no
        preference), and r_ii is 1 by definition.

        The usual binary ratios map as r_1 = r_12 and r_2 = r_21, e.g. styrene-MMA:

            SequenceMayoLewis(feed={'S': 1, 'M': 1}, reactivity={('S', 'M'): 0.52, ('M', 'S'): 0.46})

    Notes
    -----
    The feed is fixed ("instantaneous"): every chain is drawn as if formed at the same
    moment, as at low conversion. Composition drift, as monomers are used up at
    different rates, is not modelled.
    """

    def __init__(
        self,
        feed: Mapping[UnitLabel, float],
        reactivity: Mapping[tuple[UnitLabel, UnitLabel], float] | None = None,
    ) -> None:
        self.feed = _normalized(feed)
        self._labels = list(self.feed)
        index = {label: i for i, label in enumerate(self._labels)}

        n = len(self._labels)
        self.reactivity = np.ones((n, n), dtype=float)
        for (label_i, label_j), r_ij in (reactivity or {}).items():
            if (label_i not in index) or (label_j not in index):
                raise ValueError(f"Reactivity pair {(label_i, label_j)} names a unit not in the feed")
            if r_ij <= 0:
                raise ValueError(
                    f"Reactivity ratio for {(label_i, label_j)} must be positive, got {r_ij}; "
                    "use SequenceAlternating for strictly alternating sequences"
                )
            if label_i == label_j and r_ij != 1:
                raise ValueError(f"Self reactivity ratio r_ii must be 1, got {r_ij} for {label_i!r}")
            self.reactivity[index[label_i], index[label_j]] = r_ij

        # Transition probabilities P(i -> j), row i for a chain ending in unit i.
        #
        # A chain ending in i adds j at rate k_ij [M_j]. Writing k_ij = k_ii / r_ij and
        # [M_j] in proportion to f_j, that rate is k_ii * f_j / r_ij. k_ii is the same for
        # every j in row i, so it cancels when the row is normalized:
        #
        #     P(i -> j) = (f_j / r_ij) / sum_k (f_k / r_ik)
        #
        # Only ratios appear, which is why the model takes reactivity ratios (what's
        # measured and tabulated) and never needs absolute rate constants.
        weights = np.array(list(self.feed.values())) / self.reactivity  # weights[i, j] = f_j / r_ij
        self.transition = weights / weights.sum(axis=1, keepdims=True)

        # The first unit has no predecessor to condition on. Drawing it from the stationary
        # distribution pi (pi P = pi) makes every position along the chain statistically
        # alike, as if the chain were cut from the middle of a very long one. Starting from
        # the raw feed instead would bias the chain's start toward feed composition, which
        # differs from the copolymer's composition whenever the ratios aren't all 1.
        #
        # pi is the left eigenvector of P with eigenvalue 1, i.e. an eigenvector of P.T.
        # P is a stochastic matrix, so eigenvalue 1 always exists. Every r_ij is positive, so
        # each fed unit (f_j > 0) is reachable from every unit and pi is unique (an unfed
        # unit just gets pi_j = 0). argmin picks that eigenvalue robustly against round-off.
        eigvals, eigvecs = np.linalg.eig(self.transition.T)
        stationary = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
        self.stationary = stationary / stationary.sum()

    @property
    def labels(self) -> frozenset[UnitLabel]:
        return frozenset(self._labels)

    def sample(self, n_units: int, rng: np.random.Generator) -> list[UnitLabel]:
        # walk the Markov chain: the first unit from the stationary distribution, then each
        # next unit from the transition row of the one before it (one draw per unit)
        idxs = [rng.choice(len(self._labels), p=self.stationary)]
        for _ in range(n_units - 1):
            idxs.append(rng.choice(len(self._labels), p=self.transition[idxs[-1]]))
        return [self._labels[i] for i in idxs]

    def __repr__(self) -> str:
        return f"SequenceMayoLewis(feed={self.feed}, reactivity={self.reactivity.tolist()})"


class SequenceAlternating(SequenceModel):
    """Repeat the given labels cyclically, optionally starting at a random point in the cycle"""

    def __init__(self, labels: Sequence[UnitLabel], random_start: bool = False) -> None:
        if len(labels) < 2:
            raise ValueError("Alternating sequences need at least 2 labels")
        self.cycle = tuple(labels)
        self.random_start = random_start

    @property
    def labels(self) -> frozenset[UnitLabel]:
        return frozenset(self.cycle)

    def sample(self, n_units: int, rng: np.random.Generator) -> list[UnitLabel]:
        offset = int(rng.integers(len(self.cycle))) if self.random_start else 0
        return [self.cycle[(offset + i) % len(self.cycle)] for i in range(n_units)]

    def __repr__(self) -> str:
        return f"SequenceAlternating({list(self.cycle)}, random_start={self.random_start})"


class SequenceBlock(SequenceModel):
    """Contiguous blocks of single unit labels, sized as fractions of the chain"""

    def __init__(self, blocks: Sequence[tuple[UnitLabel, float]]) -> None:
        total = sum(fraction for _, fraction in blocks)
        if (not blocks) or any(fraction <= 0 for _, fraction in blocks):
            raise ValueError("Blocks must be non-empty with positive fractions")
        self.blocks = [(label, fraction / total) for label, fraction in blocks]

    @property
    def labels(self) -> frozenset[UnitLabel]:
        return frozenset(label for label, _ in self.blocks)

    def sample(self, n_units: int, rng: np.random.Generator) -> list[UnitLabel]:
        # largest-remainder rounding, so block lengths sum exactly to n_units
        exact = np.array([fraction * n_units for _, fraction in self.blocks])
        lengths = np.floor(exact).astype(int)
        for i in np.argsort(-(exact - lengths))[: n_units - lengths.sum()]:
            lengths[i] += 1
        if (lengths == 0).any():
            raise ValueError(f"Chain of {n_units} units is too short for blocks {self.blocks}")

        return [label for (label, _), length in zip(self.blocks, lengths) for _ in range(length)]

    def __repr__(self) -> str:
        return f"SequenceBlock({self.blocks})"


class SequenceExplicit(SequenceModel):
    """A fixed, user-given sequence; chain sizes must match its length"""

    def __init__(self, sequence: Sequence[UnitLabel]) -> None:
        self.sequence = list(sequence)

    @property
    def labels(self) -> frozenset[UnitLabel]:
        return frozenset(self.sequence)

    def sample(self, n_units: int, rng: np.random.Generator) -> list[UnitLabel]:
        if n_units != len(self.sequence):
            raise ValueError(
                f"Explicit sequence has {len(self.sequence)} units but {n_units} were requested; "
                f"use size=SizeMonodisperse({len(self.sequence)})"
            )
        return list(self.sequence)

    def __repr__(self) -> str:
        return f"SequenceExplicit({self.sequence})"
