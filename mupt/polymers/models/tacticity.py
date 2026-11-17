"""
Tacticity: statistical models of relative backbone configuration

Each tactic unit gets a config of +1 or -1 (the RDKit-level construction of the matching
stereo variants lives in mupt.polymers.utils.stereo). Two ADJACENT tactic units form a
diad: meso ('m') if their configs match, racemo ('r') otherwise.

In copolymers, tactic units can be separated by non-tactic ones (e.g. styrene units in
an ethylene-styrene copolymer). Models are told the size of each such gap:
    gaps[i] = number of non-tactic units between tactic unit i-1 and tactic unit i
(gaps[0] is ignored; 0 means adjacent). What a gap does depends on the mechanism:
    - isoselective site control: nothing; the catalyst sets every config independently
    - syndioselective site control: the site switches between its two coordination
      positions at EVERY insertion, non-tactic ones included, so a gap of g units
      means g + 1 chances for the site to have switched
    - chain-end control: after a non-tactic unit, the chain end has no stereocenter to
      steer the next one, so by default the correlation is lost ('reset'); 'carry'
      treats gapped tactic units as if they were adjacent
For one configuration throughout a chain regardless of gaps, use TacticityConstant.

Configs mean the same thing for every unit: +1 puts a unit's side group on one fixed
side of the backbone (see utils.stereo), so equal configs are "same side" even
between different monomers.

Random numbers: each model draws a fixed number of random numbers per call plus a
fixed number per tactic unit, never depending on the gaps, so gaps never shift the
random stream seen by later chains.

Sources (see agentic/api_v2_plan.md, section 14, for verification status):
    - definitions (tacticity, meso/racemo diads): IUPAC, Pure Appl. Chem. 1981, 53, 733
      doi:10.1351/pac198153030733
    - Bernoullian chain-end statistics: Bovey & Tiers, J. Polym. Sci. 1960, 44, 173
      doi:10.1002/pol.1960.1204414315
    - first-order Markov statistics: Coleman & Fox, J. Polym. Sci. A 1963, 1, 3183
      doi:10.1002/pol.1963.100011013
    - enantiomorphic-site model: Shelden, Fueno, Tsunetsugu & Furukawa, J. Polym. Sci. B
      1965, 3, 23  doi:10.1002/pol.1965.110030107
    - syndioselective (Cs) sites: Ewen, Jones, Razavi & Ferrara, J. Am. Chem. Soc. 1988,
      110, 6255  doi:10.1021/ja00226a056
    - polypropylene microstructure and error signatures: Busico & Cipullo, Prog. Polym.
      Sci. 2001, 26, 443  doi:10.1016/S0079-6700(00)00046-0

Not covered (extension points):
    - cotacticity: diads between DIFFERENT tactic units (e.g. styrene-MMA) are drawn with
      the same statistics as diads between identical ones; a model could be given the
      unit labels alongside the gaps and condition its statistics on each pair
    - two-site (mixture) models, e.g. heterogeneous Ziegler-Natta catalysts: approximate
      by blending the output of two generators (TopologyGenerator.realize(universe=...))
    - chain-end and site control combined within one chain: a new TacticityModel
    - hemi-isotactic (C1-symmetric) sites: a different sigma for each site state
"""

from typing import Literal, Optional, Sequence
from abc import ABC, abstractmethod

import numpy as np

from .._common import Config

type GapsLike = int | Sequence[int]  # an int n means n adjacent tactic units
type AcrossGaps = Literal["reset", "carry"]
type SiteHandedness = Literal["racemic", 1, -1]

_ACROSS_GAPS_OPTIONS = ("reset", "carry")
_SITE_HANDEDNESS_OPTIONS = ("racemic", 1, -1)


# Gaps between tactic units
def gaps_between(is_tactic: Sequence[bool]) -> list[int]:
    """
    Gaps before each tactic unit of a chain, given which units are tactic

    E.g. tactic flags for E-S-E-E-S-S give [1, 2, 0]: one gap entry per tactic unit
    """
    gaps: list[int] = []
    run = 0  # non-tactic units seen since the last tactic one
    for tactic in is_tactic:
        if tactic:
            gaps.append(run)
            run = 0
        else:
            run += 1

    return gaps


def _as_gaps(gaps: GapsLike) -> list[int]:
    """Coerce an int (that many adjacent tactic units) or a sequence of gaps to a list"""
    if isinstance(gaps, (int, np.integer)):
        if gaps < 0:
            raise ValueError(f"Number of tactic units must be non-negative, got {gaps}")
        return [0] * int(gaps)

    gaps = [int(gap) for gap in gaps]
    if any(gap < 0 for gap in gaps):
        raise ValueError(f"Gaps must be non-negative, got {gaps}")
    return gaps


# Statistical models
class TacticityModel(ABC):
    """Abstract base class for drawing configurations of the tactic units in a chain"""

    @abstractmethod
    def sample(self, gaps: GapsLike, rng: np.random.Generator) -> list[Config]:
        """
        Draw one config (+1 or -1) per tactic unit, in chain order

        gaps is either the number of tactic units (all adjacent), or the gap before
        each one (see gaps_between). The number of random numbers drawn must not
        depend on the gaps (see module docstring)
        """
        ...


def _check_probability(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _check_across_gaps(across_gaps: str) -> None:
    if across_gaps not in _ACROSS_GAPS_OPTIONS:
        raise ValueError(f"across_gaps must be one of {_ACROSS_GAPS_OPTIONS}, got {across_gaps!r}")


def _random_config(draw: float) -> Config:
    """An unbiased config from one uniform draw"""
    return 1 if (draw < 0.5) else -1


class TacticityBernoullian(TacticityModel):
    """
    Chain-end control: each diad is meso with probability p_m, independently

    across_gaps='reset' (default) starts a fresh, unbiased config after any gap;
    'carry' treats tactic units separated by a gap as if they were adjacent
    """

    def __init__(self, p_m: float, across_gaps: AcrossGaps = "reset") -> None:
        _check_probability("p_m", p_m)
        _check_across_gaps(across_gaps)
        self.p_m = p_m
        self.across_gaps = across_gaps

    def sample(self, gaps: GapsLike, rng: np.random.Generator) -> list[Config]:
        gaps = _as_gaps(gaps)
        draws = rng.random(len(gaps))  # one per tactic unit

        configs: list[Config] = []
        for i, (gap, draw) in enumerate(zip(gaps, draws)):
            if (i == 0) or (gap > 0 and self.across_gaps == "reset"):
                configs.append(_random_config(draw))  # start of a new stereo run
            else:
                is_meso = draw < self.p_m
                configs.append(configs[-1] if is_meso else -configs[-1])

        return configs

    def __repr__(self) -> str:
        return f"TacticityBernoullian(p_m={self.p_m}, across_gaps={self.across_gaps!r})"


class TacticityMarkov(TacticityModel):
    """
    Chain-end control with memory: each diad's type depends on the previous diad

    P(m | previous m) and P(m | previous r) are given; the first diad of each run is
    drawn from the stationary distribution. across_gaps behaves as for TacticityBernoullian,
    and a reset also forgets the previous diad
    """

    def __init__(self, p_m_given_m: float, p_m_given_r: float, across_gaps: AcrossGaps = "reset") -> None:
        _check_probability("p_m_given_m", p_m_given_m)
        _check_probability("p_m_given_r", p_m_given_r)
        _check_across_gaps(across_gaps)
        self.p_m_given_m = p_m_given_m
        self.p_m_given_r = p_m_given_r
        self.across_gaps = across_gaps

    @property
    def stationary_meso_probability(self) -> float:
        """Long-run fraction of meso diads, used for the first diad of each run"""
        denom = 1.0 - self.p_m_given_m + self.p_m_given_r
        return (self.p_m_given_r / denom) if (denom > 0) else 1.0

    def sample(self, gaps: GapsLike, rng: np.random.Generator) -> list[Config]:
        gaps = _as_gaps(gaps)
        draws = rng.random(len(gaps))  # one per tactic unit

        configs: list[Config] = []
        previous_meso: Optional[bool] = None  # None: no diad yet in this run
        for i, (gap, draw) in enumerate(zip(gaps, draws)):
            if (i == 0) or (gap > 0 and self.across_gaps == "reset"):
                configs.append(_random_config(draw))
                previous_meso = None
                continue

            if previous_meso is None:
                p_m = self.stationary_meso_probability
            else:
                p_m = self.p_m_given_m if previous_meso else self.p_m_given_r

            previous_meso = draw < p_m
            configs.append(configs[-1] if previous_meso else -configs[-1])

        return configs

    def __repr__(self) -> str:
        return (
            f"TacticityMarkov(p_m_given_m={self.p_m_given_m}, p_m_given_r={self.p_m_given_r}, "
            f"across_gaps={self.across_gaps!r})"
        )


class TacticityEnantiomorphicSite(TacticityModel):
    """
    Catalyst-site control (enantiomorphic-site model): the catalyst site sets each config

    The site has a state s = +1 or -1, the config it currently favors. At each insertion
    the new unit takes config s with probability sigma (the site's selectivity) and -s
    otherwise; errors aren't remembered. After each insertion the site switches state
    with probability p_flip:
        - p_flip = 0 (default): isoselective sites (e.g. C2-symmetric metallocenes)
        - p_flip near 1: syndioselective sites (e.g. Cs-symmetric metallocenes), whose
          two coordination positions favor opposite faces; see syndiospecific_site()

    Parameters
    ----------
    sigma : float in [1/2, 1]
        Selectivity: probability that an insertion follows the site's preference
    p_flip : float in [0, 1], default 0
        Probability that the site switches state after an insertion (any insertion,
        including non-tactic comonomers in a gap)
    site_handedness : 'racemic', +1, or -1, default 'racemic'
        Each chain's starting site state: drawn at random per chain for a racemic
        catalyst, or fixed (an enantiopure catalyst)
    """

    def __init__(
        self,
        sigma: float,
        p_flip: float = 0.0,
        site_handedness: SiteHandedness = "racemic",
    ) -> None:
        if not (0.5 <= sigma <= 1.0):
            raise ValueError(
                f"sigma is the site's selectivity and must be in [1/2, 1], got {sigma}; "
                "to favor the opposite config, set site_handedness instead"
            )
        _check_probability("p_flip", p_flip)
        if site_handedness not in _SITE_HANDEDNESS_OPTIONS:
            raise ValueError(f"site_handedness must be one of {_SITE_HANDEDNESS_OPTIONS}, got {site_handedness!r}")

        self.sigma = sigma
        self.p_flip = p_flip
        self.site_handedness = site_handedness

    def net_flip_probability(self, n_insertions: int) -> float:
        """Probability that the site has switched state an odd number of times over n insertions"""
        return 0.5 * (1.0 - (1.0 - 2.0 * self.p_flip) ** n_insertions)

    def sample(self, gaps: GapsLike, rng: np.random.Generator) -> list[Config]:
        gaps = _as_gaps(gaps)

        # random numbers: one per chain for a racemic site's handedness, then one per tactic
        # unit for the face it takes, then (if the site can switch) one per tactic unit for that
        state = _random_config(rng.random()) if (self.site_handedness == "racemic") else self.site_handedness
        face_draws = rng.random(len(gaps))
        flip_draws = rng.random(len(gaps)) if (self.p_flip > 0) else None

        configs: list[Config] = []
        for i, gap in enumerate(gaps):
            # since the previous tactic unit: its own insertion plus one per non-tactic unit
            if (i > 0) and (flip_draws is not None):
                if flip_draws[i] < self.net_flip_probability(gap + 1):
                    state = -state

            configs.append(state if (face_draws[i] < self.sigma) else -state)

        return configs

    def __repr__(self) -> str:
        return (
            f"TacticityEnantiomorphicSite(sigma={self.sigma}, p_flip={self.p_flip}, "
            f"site_handedness={self.site_handedness!r})"
        )


class TacticityConstant(TacticityModel):
    """
    Every tactic unit of a chain gets the same config, whatever the gaps between them

    With config=None, each chain's config is chosen at random (+1 or -1); otherwise
    every chain gets the given config
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        if config not in (None, 1, -1):
            raise ValueError(f"Config must be +1, -1, or None (random per chain), got {config!r}")
        self.config = config

    def sample(self, gaps: GapsLike, rng: np.random.Generator) -> list[Config]:
        draws = rng.random(len(_as_gaps(gaps)))  # one per tactic unit, like every model
        if len(draws) == 0:
            return []

        config = self.config if (self.config is not None) else _random_config(draws[0])
        return [config] * len(draws)

    def __repr__(self) -> str:
        return f"TacticityConstant(config={self.config})"


# Presets
def isotactic(across_gaps: AcrossGaps = "reset") -> TacticityBernoullian:
    """
    Chain-end controlled isotactic: every diad between ADJACENT tactic units is meso

    With the default across_gaps='reset', a run of tactic units after a non-tactic one
    starts from a random config, so in a copolymer separate runs may be inverted
    relative to each other. Use across_gaps='carry', or TacticityConstant, for one config
    throughout the chain.
    """
    return TacticityBernoullian(1.0, across_gaps=across_gaps)


def syndiotactic(across_gaps: AcrossGaps = "reset") -> TacticityBernoullian:
    """
    Chain-end controlled syndiotactic: every diad between ADJACENT tactic units is racemo

    With the default across_gaps='reset', a run of tactic units after a non-tactic one
    starts from a random config, so the alternation isn't kept across gaps. Use
    across_gaps='carry' to keep alternating through them.
    """
    return TacticityBernoullian(0.0, across_gaps=across_gaps)


def atactic() -> TacticityBernoullian:
    """Random relative configurations: every diad is meso with probability 1/2"""
    return TacticityBernoullian(0.5)


def syndiospecific_site(
    sigma: float = 1.0,
    p_skip: float = 0.0,
    site_handedness: SiteHandedness = "racemic",
) -> TacticityEnantiomorphicSite:
    """
    Syndioselective catalyst-site control (e.g. Cs-symmetric metallocenes)

    The site switches between its two coordination positions, which favor opposite faces,
    at every insertion. sigma is the facial selectivity: a facial error gives an isolated
    inverted unit (an rmmr pentad). p_skip is the probability that the site does NOT switch
    between two insertions, i.e. an odd number of "back-skips" (site epimerizations):
    a skip gives an isolated meso diad (an rmrr pentad)
    """
    _check_probability("p_skip", p_skip)
    return TacticityEnantiomorphicSite(sigma, p_flip=1.0 - p_skip, site_handedness=site_handedness)



# Diad statistics
def configs_to_diads(
    configs: Sequence[Config],
    gaps: Optional[Sequence[int]] = None,
    *,
    include_gapped: bool = False,
) -> list[Literal["m", "r"]]:
    """
    Classify pairs of consecutive tactic units as meso ('m') or racemo ('r')

    Without gaps, all units are taken as adjacent. With gaps, only truly adjacent
    pairs count as diads (as measured experimentally), unless include_gapped is set
    """
    if (gaps is not None) and (len(gaps) != len(configs)):
        raise ValueError(f"Need one gap per config, got {len(gaps)} gaps for {len(configs)} configs")

    diads: list[Literal["m", "r"]] = []
    for i in range(1, len(configs)):
        if (gaps is not None) and (gaps[i] > 0) and not include_gapped:
            continue
        diads.append("m" if configs[i] == configs[i - 1] else "r")

    return diads


def meso_fraction(
    configs: Sequence[Config],
    gaps: Optional[Sequence[int]] = None,
    *,
    include_gapped: bool = False,
) -> float:
    """Fraction of diads which are meso (NaN if there are none); see configs_to_diads"""
    diads = configs_to_diads(configs, gaps, include_gapped=include_gapped)
    return diads.count("m") / len(diads) if diads else float("nan")
