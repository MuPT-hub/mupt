"""
Generation of polymer systems: chain sizes, sequences, tacticity, and topology realization

Subpackages
-----------
models
    Statistical models: sequence, tacticity, and chain-size distributions
rules
    What may bond to what, and how: faces, junction rules, port compatibility
utils
    Unit and graph machinery: lexicons, stereo, realization, composition
generators
    Topology generators producing descriptive graphs of polymer chains

Every public name of the subpackages is also re-exported here, e.g.
`from mupt.polymers import LinearPolymerGenerator, RandomFlip, SizeSchulzZimm`.
"""

from .exceptions import (
    FaceError as FaceError,
    TopologyValidationError as TopologyValidationError,
    CompatibilityError as CompatibilityError,
)
from ._common import FixedFace as FixedFace

# statistical models
from .models import (
    SizeDistribution as SizeDistribution,
    SizeMonodisperse as SizeMonodisperse,
    SizeUniform as SizeUniform,
    SizeSchulzZimm as SizeSchulzZimm,
    SizeFlory as SizeFlory,
    coerce_size as coerce_size,
    SequenceModel as SequenceModel,
    SequenceHomopolymer as SequenceHomopolymer,
    SequenceRandom as SequenceRandom,
    SequenceMayoLewis as SequenceMayoLewis,
    SequenceAlternating as SequenceAlternating,
    SequenceBlock as SequenceBlock,
    SequenceExplicit as SequenceExplicit,
    TacticityModel as TacticityModel,
    TacticityBernoullian as TacticityBernoullian,
    TacticityMarkov as TacticityMarkov,
    TacticityEnantiomorphicSite as TacticityEnantiomorphicSite,
    TacticityConstant as TacticityConstant,
    isotactic as isotactic,
    syndiotactic as syndiotactic,
    atactic as atactic,
    syndiospecific_site as syndiospecific_site,
    configs_to_diads as configs_to_diads,
    gaps_between as gaps_between,
    meso_fraction as meso_fraction,
)

# rules
from .rules import (
    normalize_faces as normalize_faces,
    validate_faces as validate_faces,
    JunctionRule as JunctionRule,
    FixedPairing as FixedPairing,
    RandomFlip as RandomFlip,
    ANY as ANY,
    CompatibilityRule as CompatibilityRule,
    CompatibleSites as CompatibleSites,
    head_to_tail as head_to_tail,
    free_faces as free_faces,
    resolve_pattern as resolve_pattern,
)

# graph machinery
from .utils import (
    primitive_from_descriptive_graph as primitive_from_descriptive_graph,
    universe_from_descriptive_graphs as universe_from_descriptive_graphs,
    combine_universes as combine_universes,
    validate_descriptive_graph as validate_descriptive_graph,
    resolve_component_graph as resolve_component_graph,
    flatten_component_graph as flatten_component_graph,
    validate_component_graph as validate_component_graph,
    draw_topology as draw_topology,
    draw_component_graph as draw_component_graph,
)

# generators
from .generators import (
    TopologyGenerator as TopologyGenerator,
    CompositeGenerator as CompositeGenerator,
    LinearPolymerGenerator as LinearPolymerGenerator,
    LadderPolymerGenerator as LadderPolymerGenerator,
    UnitGenerator as UnitGenerator,
    BlockCopolymerGenerator as BlockCopolymerGenerator,
    StarPolymerGenerator as StarPolymerGenerator,
)
