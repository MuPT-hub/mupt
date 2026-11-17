"""
Distributions of chain lengths for polymer topology generation

Sizes are degrees of polymerization (DP) counted in repeat units, EXCLUDING
end groups; dispersities are on a DP (number) basis, not a mass basis
"""
# DEV: a quick and dirty implementation needs polish
# TODO: polish
from typing import Any, Iterable
from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

# typing
from ...geometry.arraytypes import Shape, N
IntVectorN = np.ndarray[Shape[N], np.dtype[np.integer]]

class SizeDistribution(ABC):
    """
    Abstract base class for distributions of polymer chain sizes
    """

    # Sampling
    def sample(self, n_chains: int, rng: np.random.Generator) -> list[int]:
        """
        Draw degrees of polymerization for a collection of chains

        Parameters
        ----------
        n_chains : int
            The number of chain sizes to draw; must be at least 1
        rng : np.random.Generator
            Source of randomness; deterministic distributions may ignore this,
            but must not draw from numpy's global random state

        Returns
        -------
        sizes : list[int]
            One degree of polymerization (>= 1) per chain, as native Python ints
        """
        # bool is a subclass of int, so must be excluded explicitly
        if isinstance(n_chains, bool) or not isinstance(n_chains, (int, np.integer)):
            raise TypeError(f"n_chains must be an integer, is {type(n_chains).__name__}")
        if n_chains < 1:
            raise ValueError(f"n_chains must be at least 1, is {n_chains}")

        # run sampling
        n_chains = int(n_chains)
        chain_lengths = np.asarray(self._sample(n_chains, rng))

        # validate output
        if chain_lengths.shape != (n_chains,):
            raise RuntimeError(
                f"{self!r} produced sizes of shape {chain_lengths.shape}, expected ({n_chains},)"
            )
        if not np.issubdtype(chain_lengths.dtype, np.integer):
            raise RuntimeError(
                f"{self!r} produced non-integer sizes of dtype {chain_lengths.dtype}"
            )
        if (chain_lengths < 1).any():
            raise RuntimeError(f"{self!r} produced sizes smaller than 1: {chain_lengths.min()}")

        return [int(size) for size in chain_lengths]  # coerce away numpy scalar types

    @abstractmethod
    def _sample(self, n_chains: int, rng: np.random.Generator) -> np.ndarray:
        """
        Implement drawing of chain sizes here

        Should return an integer array of shape (n_chains,) with all entries >= 1;
        n_chains is guaranteed to be a validated positive int
        """
        ...

class SizeMonodisperse(SizeDistribution):
    """Every chain has exactly DPn units"""

    def __init__(
            self,
            DPn: int,
            ) -> None:
        if DPn < 1:
            raise ValueError(f'DPn must be >= 1, is {DPn}')
        self.DPn = DPn

    def __repr__(self) -> str:
        return f'SizeMonodisperse(DPn={self.DPn})'

    def _sample(self, n_chains: int, rng: Generator | None = None) -> np.ndarray:
        return np.full(n_chains, self.DPn)

class SizeUniform(SizeDistribution):
    """Chain sizes drawn uniformly from [min_dp, max_dp], inclusive on both ends"""

    def __init__(
            self,
            min_dp: int,
            max_dp: int,
            ) -> None:
        if not (1 <= min_dp <= max_dp):
            raise ValueError(f'Require 1 <= min_dp <= max_dp, got {min_dp}, {max_dp}')
        self.min_dp = min_dp
        self.max_dp = max_dp

    def __repr__(self) -> str:
        return f'SizeUniform(min_dp={self.min_dp}, max_dp={self.max_dp})'

    def _sample(self, n_chains: int, rng: Generator) -> np.ndarray:
        return rng.integers(
            size=n_chains,
            low=self.min_dp, 
            high=self.max_dp, 
            endpoint=True,
        )

class SizeSchulzZimm(SizeDistribution):
    """
    Schulz-Zimm (gamma) number distribution with number-average DPn and dispersity PDI

    Sizes are rounded and clipped to >= 1, which biases the realized mean and PDI at small DPn
    """

    def __init__(
            self,
            DPn: float,
            PDI: float,
            ) -> None:
        if DPn < 1:
            raise ValueError(
                f'DPn must be >= 1, is {DPn}'
            )
        if PDI <= 1:
            raise ValueError(
                f'PDI must be > 1, is {PDI}, use MonoDisperse if PDI == 1'
            )
        
        self.DPn = DPn
        self.PDI = PDI

    def _sample(self, n_chains: int, rng: Generator) -> np.ndarray:
        shape = 1.0 / (self.PDI - 1.0)
        scale = self.DPn / shape

        chain_lengths = rng.gamma(shape=shape, scale=scale, size=n_chains)
        return np.clip(chain_lengths.round(0), 1, None).astype(int)

    def __repr__(self) -> str:
        return f'SizeSchulzZimm(DPn={self.DPn}, PDI={self.PDI})'


class SizeFlory(SizeDistribution):
    """Flory (most-probable, geometric) number distribution with number-average DPn"""

    def __init__(
            self,
            DPn: int,
            ) -> None:
        if DPn < 1:
            raise ValueError(
                f'DPn must be >= 1, is {DPn}'
            )
        
        self.DPn = DPn

    def _sample(self, n_chains: int, rng: Generator) -> np.ndarray:
        return rng.geometric(1/self.DPn, n_chains)

    def __repr__(self) -> str:
        return f'SizeFlory(DPn={self.DPn})'


def coerce_size(size: SizeDistribution | int) -> SizeDistribution:
    """Interpret an int as SizeMonodisperse; pass SizeDistributions through unchanged"""
    if isinstance(size, SizeDistribution):
        return size
    if isinstance(size, (int, np.integer)) and not isinstance(size, bool):
        return SizeMonodisperse(int(size))
    raise TypeError(
        f'size must be a SizeDistribution or int, not {type(size).__name__}'
    )
