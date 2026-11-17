from types import MappingProxyType
from typing import Mapping, Optional
import warnings

import numpy as np
from networkx import Graph

from .._common import UnitLabel
from ..exceptions import FaceError
from ..models.polydispersity import SizeDistribution
from ..models.sequence import SequenceModel
from ..models.tacticity import TacticityModel
from ..rules.faces import face_anchor_indices
from ..rules.junctions import FixedPairing, JunctionRule, anchors_symmetry_equivalent
from ..utils.lexicon import UnitLike, unit_mol

from .base import TopologyGenerator
from .linear import EndGroup, _ChainTopologyGenerator


#region Ladder polymers
class LadderPolymerGenerator(_ChainTopologyGenerator):
    """
    Generator of ladder polymer chains, whose units are joined by two or more bonds

    Each unit's 'tail' face bonds to the next unit's 'head' face, strand by strand,
    with node 0 at the head end. A junction rule decides how the strands pair up
    (e.g. RandomFlip for PIM-1's two linkage patterns). For chains joined by single
    bonds, use LinearPolymerGenerator.

    Parameters
    ----------
    units : Mapping[UnitLabel, str | Primitive]
        Repeat units, as SMILES or Primitives, with ports labelled by atom map numbers
    size : SizeDistribution | int
        Number of repeat units per chain (end groups excluded); an int means every
        chain has exactly that many
    faces : Mapping[UnitLabel, {'head': ports, 'tail': ports}]
        Required: each unit's head and tail port labels, in strand order, e.g.
        {'TTSBI': {'head': (1, 2), 'tail': (3, 4)}}; every face must have the same width
    end_groups : str | Primitive | {'head': ..., 'tail': ...}, optional
        Caps for the chain ends; a single value caps both ends, None leaves ends free.
        A single-port cap is attached once per strand (which makes the end unit a
        branch point); a cap with one port per strand is attached once. A cap given as
        {label: cap} is named, and its faces may be given in faces (see LinearPolymerGenerator)
    sequence : SequenceModel, optional
        Unit sequence statistics; defaults to an equimolar SequenceRandom
    tacticity : TacticityModel, optional
        Configurations of units with backbone stereocenters (e.g. TTSBI's spiro carbon).
        Step-growth ladders such as PIM-1 have no chain-end control, so site-type
        models (TacticityEnantiomorphicSite, e.g. racemic) or TacticityConstant usually fit better
        than chain-end ones (TacticityBernoullian, isotactic, ...)
    side_groups : Mapping[UnitLabel, int], optional
        Atom map number of the side group used to define a unit's configuration; only
        needed to override the default
    junctions : JunctionRule, optional
        How the strands pair at each junction; defaults to FixedPairing (straight)
    embed_random_seed : int, default 0
        Seed for embedding unit coordinates (-1 for non-deterministic)
    """

    def __init__(
        self,
        units: Mapping[UnitLabel, UnitLike],
        *,
        size: SizeDistribution | int,
        faces: Mapping[UnitLabel, Mapping],
        end_groups: Optional[EndGroup | Mapping[str, Optional[EndGroup]]] = None,
        sequence: Optional[SequenceModel] = None,
        tacticity: Optional[TacticityModel] = None,
        side_groups: Optional[Mapping[UnitLabel, int]] = None,
        junctions: Optional[JunctionRule] = None,
        embed_random_seed: int = 0,
    ) -> None:
        super().__init__(
            units,
            size=size,
            faces=faces,
            end_groups=end_groups,
            sequence=sequence,
            tacticity=tacticity,
            side_groups=side_groups,
            junctions=junctions,
            embed_random_seed=embed_random_seed,
        )
        self._warn_degenerate_flips(units)

    def _check_width(self, width: int) -> None:
        if width < 2:
            raise FaceError(
                f"LadderPolymerGenerator builds chains joined by several bonds per junction, "
                f"but the faces give {width}; use LinearPolymerGenerator instead"
            )

    def _warn_degenerate_flips(self, units: Mapping[UnitLabel, UnitLike]) -> None:
        """Warn when a junction rule can flip a face whose ports are symmetry-equivalent"""
        if isinstance(self.junctions, FixedPairing):
            return  # nothing is ever flipped

        for label, unit in units.items():
            mol = unit_mol(unit)
            for side in ("head", "tail"):
                anchors = face_anchor_indices(mol, self._unit_faces[label], side)
                if anchors_symmetry_equivalent(mol, anchors):
                    warnings.warn(
                        f"The {side} ports of unit {label!r} are symmetry-equivalent, so flipping "
                        f"junctions at that face may not produce a distinct molecule"
                    )
