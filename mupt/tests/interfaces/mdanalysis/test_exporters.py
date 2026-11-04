"""
Tests to ensure export from MuPT to MDAnalysis
preserves molecular identity and connectivity.
"""

# Shortcut to run tests for this file:
# python -m pytest mupt/tests/interfaces/mdanalysis/test_exporters.py -v
from collections import Counter

import numpy as np
import pytest

from mupt.chemistry import ELEMENTS
from mupt.chemistry.core import BondType
from mupt.graphs.generators import path_graph_from_sequence
from mupt.interfaces.mdanalysis import (
    AllAtomExportStrategy,
    CoarseGrainedExportStrategy,
    primitive_to_mdanalysis,
)
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.mupr.connection.connectors import AttachmentPoint, Connector
from mupt.mupr.primitives import (
    AtomicPrimitive,
    CompositePrimitive,
    RootPrimitive,
    SimplePrimitive,
)
from mupt.mupr.primitives.predicates import is_atom
from mupt.roles import PrimitiveRole


# ============================================================================
# FIXTURE HELPERS
# ============================================================================


def build_lexicon(smiles: dict[str, str], **kwargs) -> dict[str, CompositePrimitive]:
    """One repeat-unit Primitive per SMILES, labelled by its lexicon key."""
    return {
        name: primitive_from_smiles(smi, label=name, **kwargs)
        for name, smi in smiles.items()
    }


def build_chain(
    lexicon: dict[str, CompositePrimitive],
    sequence: list[str],
    label: str = "chain",
) -> CompositePrimitive:
    """Link copies of lexicon entries along a linear sequence."""
    chain = CompositePrimitive(label=label)
    chain.populate_from_topology_and_lexicon(
        path_graph_from_sequence(sequence),
        lexicon,
        set_label_from_lexicon=True,
    )
    return chain


def tag_roles(root: RootPrimitive) -> RootPrimitive:
    """Tag Root -> chain -> repeat unit: chains as segments, units as residues.

    Roles are set after building because ``copy()``, which
    ``populate_from_topology_and_lexicon`` uses, does not carry them over yet.
    """
    for chain in root.children:
        chain.role = "segment"
        for unit in chain.children:
            unit.role = "residue"
    return root


def polyethylene_sequence(n_units: int) -> list[str]:
    """Sequence of head, (n - 2) ethane and tail repeat units."""
    return ["head"] + (n_units - 2) * ["ethane"] + ["tail"]


def count_atoms(root: RootPrimitive) -> int:
    """Number of atoms in a MuPT hierarchy."""
    return root.cross_section(is_atom).number_of_nodes()


def count_bonds(root: RootPrimitive) -> int:
    """Number of distinct bonded atom pairs in a MuPT hierarchy."""
    return len({frozenset(edge) for edge in root.cross_section(is_atom).edges()})


def single_molecule(unit: CompositePrimitive) -> RootPrimitive:
    """Root -> molecule (segment) -> one repeat unit (residue)."""
    molecule = CompositePrimitive(label="mol", children=[unit])
    return tag_roles(RootPrimitive(children=[molecule]))


@pytest.fixture
def polyethylene_lexicon(polyethylene_smiles):
    """Polyethylene repeat units built from SMILES."""
    return build_lexicon(polyethylene_smiles)


@pytest.fixture
def PES_lexicon(polyethersulfone_smiles):
    """Polyethersulfone repeat units built from SMILES."""
    return build_lexicon(polyethersulfone_smiles)


@pytest.fixture
def polyethylene_2mer(polyethylene_lexicon) -> RootPrimitive:
    """One polyethylene chain of 2 repeat units."""
    chain = build_chain(polyethylene_lexicon, polyethylene_sequence(2))
    return tag_roles(RootPrimitive(label="universe", children=[chain]))


@pytest.fixture
def polyethylene_3mer(polyethylene_lexicon) -> RootPrimitive:
    """One polyethylene chain of 3 repeat units."""
    chain = build_chain(polyethylene_lexicon, polyethylene_sequence(3))
    return tag_roles(RootPrimitive(label="universe", children=[chain]))


@pytest.fixture
def polyethylene_multichain(polyethylene_lexicon) -> RootPrimitive:
    """Ten chains of different lengths."""
    chains = [
        build_chain(polyethylene_lexicon, polyethylene_sequence(n), f"chain_{i}")
        for i, n in enumerate([2, 3, 4, 5, 6, 2, 3, 4, 5, 6])
    ]
    return tag_roles(RootPrimitive(label="universe", children=chains))


@pytest.fixture
def PES_copolymer(PES_lexicon) -> RootPrimitive:
    """One polyethersulfone chain of 6 repeat units."""
    sequence = ["head"] + 2 * ["bisphenol_S", "bisphenol_A"] + ["tail"]
    chain = build_chain(PES_lexicon, sequence)
    return tag_roles(RootPrimitive(label="universe", children=[chain]))


@pytest.fixture
def helium_saamr() -> RootPrimitive:
    """Root -> molecule -> repeat unit -> one He atom."""
    unit = CompositePrimitive(label="He_unit")
    unit.attach_child(AtomicPrimitive(ELEMENTS[2], label="He"))
    molecule = CompositePrimitive(label="He_molecule")
    molecule.attach_child(unit)
    return tag_roles(RootPrimitive(label="universe", children=[molecule]))


@pytest.fixture
def helium_resname_map() -> dict[str, str]:
    """Residue name for the helium repeat unit."""
    return {"He_unit": "HEL"}


# ============================================================================
# ALL-ATOM EXPORT
# ============================================================================

SYSTEMS = [
    ("polyethylene_2mer", "polyethylene_resname_map", 1, 2),
    ("polyethylene_3mer", "polyethylene_resname_map", 1, 3),
    ("polyethylene_multichain", "polyethylene_resname_map", 10, 40),
    ("PES_copolymer", "PES_resname_map", 1, 6),
]
SYSTEM_IDS = ["2mer", "3mer", "multi_chain", "PES"]


@pytest.mark.parametrize(
    "system_fixture,resname_fixture,n_segments,n_residues", SYSTEMS, ids=SYSTEM_IDS
)
def test_atom_count_preservation(
    system_fixture, resname_fixture, n_segments, n_residues, request
):
    """Every atom in the hierarchy becomes one MDAnalysis atom."""
    root = request.getfixturevalue(system_fixture)
    universe = primitive_to_mdanalysis(root, request.getfixturevalue(resname_fixture))
    assert universe.atoms.n_atoms == count_atoms(root)


@pytest.mark.parametrize(
    "system_fixture,resname_fixture,n_segments,n_residues", SYSTEMS, ids=SYSTEM_IDS
)
def test_bond_connectivity_preservation(
    system_fixture, resname_fixture, n_segments, n_residues, request
):
    """Every bonded atom pair in the hierarchy becomes one bond."""
    root = request.getfixturevalue(system_fixture)
    universe = primitive_to_mdanalysis(root, request.getfixturevalue(resname_fixture))
    assert len(universe.bonds) == count_bonds(root)


@pytest.mark.parametrize(
    "system_fixture,resname_fixture,n_segments,n_residues", SYSTEMS, ids=SYSTEM_IDS
)
def test_segment_and_residue_counts(
    system_fixture, resname_fixture, n_segments, n_residues, request
):
    """Segment-role chains become segments and residue-role units become residues."""
    root = request.getfixturevalue(system_fixture)
    universe = primitive_to_mdanalysis(root, request.getfixturevalue(resname_fixture))
    assert universe.segments.n_segments == n_segments
    assert universe.residues.n_residues == n_residues


@pytest.mark.parametrize(
    "system_fixture,resname_fixture,n_segments,n_residues", SYSTEMS, ids=SYSTEM_IDS
)
def test_atoms_are_contiguous_by_residue(
    system_fixture, resname_fixture, n_segments, n_residues, request
):
    """Each residue's atoms are listed together."""
    root = request.getfixturevalue(system_fixture)
    universe = primitive_to_mdanalysis(root, request.getfixturevalue(resname_fixture))
    assert np.all(np.diff(universe.atoms.resindices) >= 0)


def test_acyclic_chains_have_one_fewer_bond_than_atoms(
    polyethylene_multichain, polyethylene_resname_map
):
    """Linear, ring-free chains have n_atoms - 1 bonds each."""
    universe = primitive_to_mdanalysis(
        polyethylene_multichain, polyethylene_resname_map
    )
    assert len(universe.bonds) == universe.atoms.n_atoms - universe.segments.n_segments


def test_residue_ids_restart_per_segment(
    polyethylene_multichain, polyethylene_resname_map
):
    """Residue IDs count from 1 within each segment."""
    universe = primitive_to_mdanalysis(
        polyethylene_multichain, polyethylene_resname_map
    )
    for segment in universe.segments:
        n_residues = segment.residues.n_residues
        assert list(segment.residues.resids) == list(range(1, n_residues + 1))


def test_resnames_from_resname_map(polyethylene_3mer, polyethylene_resname_map):
    """Residue names are looked up from the residue labels."""
    universe = primitive_to_mdanalysis(polyethylene_3mer, polyethylene_resname_map)
    assert list(universe.residues.resnames) == ["HEA", "EAN", "TYL"]


def test_inter_residue_bonds_connect_carbons(
    polyethylene_3mer, polyethylene_resname_map
):
    """Bonds between repeat units join the backbone carbons."""
    universe = primitive_to_mdanalysis(polyethylene_3mer, polyethylene_resname_map)
    inter_residue = [
        bond for bond in universe.bonds if bond[0].resindex != bond[1].resindex
    ]
    assert len(inter_residue) == 2  # head-ethane, ethane-tail
    for bond in inter_residue:
        assert {atom.element for atom in bond} == {"C"}


def test_polyethylene_bond_orders_are_single(
    polyethylene_multichain, polyethylene_resname_map
):
    """Bond orders are carried over (all single for PE)."""
    universe = primitive_to_mdanalysis(
        polyethylene_multichain, polyethylene_resname_map
    )
    assert {bond.order for bond in universe.bonds} == {1.0}


def test_aromatic_bond_orders_preserved(PES_copolymer, PES_resname_map):
    """Aromatic and double bond orders are carried over."""
    universe = primitive_to_mdanalysis(PES_copolymer, PES_resname_map)
    orders = Counter(bond.order for bond in universe.bonds)
    assert orders[1.5] > 0  # benzene rings
    assert orders[2.0] > 0  # sulfonyl S=O


def test_element_names_and_masses(PES_copolymer, PES_resname_map):
    """Atoms get element symbols as names and elemental masses."""
    universe = primitive_to_mdanalysis(PES_copolymer, PES_resname_map)
    assert set(universe.atoms.elements) == {"C", "H", "O", "S"}
    assert list(universe.atoms.names) == list(universe.atoms.elements)
    assert universe.atoms.total_mass() == pytest.approx(
        sum(atom.mass for atom in PES_copolymer.cross_section(is_atom).nodes)
    )


def test_resname_from_metadata_overrides_resname_map(polyethylene_smiles):
    """metadata['residue_name'] takes precedence over resname_map."""
    lexicon = build_lexicon(polyethylene_smiles)
    lexicon["ethane"].metadata["residue_name"] = "PEE"
    chain = build_chain(lexicon, polyethylene_sequence(3))
    root = tag_roles(RootPrimitive(children=[chain]))
    universe = primitive_to_mdanalysis(
        root, {"head": "HEA", "ethane": "EAN", "tail": "TYL"}
    )
    assert list(universe.residues.resnames) == ["HEA", "PEE", "TYL"]


def test_invalid_resname_raises_value_error(polyethylene_2mer):
    """Residue names that are not 3 characters are rejected."""
    with pytest.raises(ValueError, match="not 3 characters"):
        primitive_to_mdanalysis(polyethylene_2mer, {"head": "TOOLONG", "tail": "TYL"})


def test_explicit_strategy_produces_same_result(
    polyethylene_3mer, polyethylene_resname_map
):
    """Passing AllAtomExportStrategy explicitly matches the default."""
    default = primitive_to_mdanalysis(polyethylene_3mer, polyethylene_resname_map)
    explicit = primitive_to_mdanalysis(
        polyethylene_3mer, polyethylene_resname_map, strategy=AllAtomExportStrategy()
    )
    assert default.atoms.n_atoms == explicit.atoms.n_atoms
    assert list(default.atoms.elements) == list(explicit.atoms.elements)
    assert default.bonds.indices.tolist() == explicit.bonds.indices.tolist()


def test_helium(helium_saamr, helium_resname_map):
    """A single-atom system exports with no bonds."""
    universe = primitive_to_mdanalysis(helium_saamr, helium_resname_map)
    assert universe.atoms.n_atoms == 1
    assert list(universe.atoms.elements) == ["He"]
    assert list(universe.residues.resnames) == ["HEL"]
    assert not hasattr(universe, "bonds")  # no bonds to add


# ============================================================================
# ROLES
# ============================================================================


def test_roles_accept_enum_members(polyethylene_lexicon, polyethylene_resname_map):
    """PrimitiveRole members work the same as their string values."""
    chain = build_chain(polyethylene_lexicon, polyethylene_sequence(3))
    chain.role = PrimitiveRole.SEGMENT
    for unit in chain.children:
        unit.role = PrimitiveRole.RESIDUE
    universe = primitive_to_mdanalysis(
        RootPrimitive(children=[chain]), polyethylene_resname_map
    )
    assert universe.segments.n_segments == 1
    assert universe.residues.n_residues == 3


def test_extra_level_above_chains(polyethylene_lexicon, polyethylene_resname_map):
    """Root -> group -> chain -> repeat unit -> atom; the untagged group is skipped."""
    chains = [
        build_chain(polyethylene_lexicon, polyethylene_sequence(n)) for n in (2, 3)
    ]
    group = CompositePrimitive(label="group", children=chains)
    root = RootPrimitive(children=[group])
    for chain in chains:
        chain.role = "segment"
        for unit in chain.children:
            unit.role = "residue"

    universe = primitive_to_mdanalysis(root, polyethylene_resname_map)
    assert universe.segments.n_segments == 2
    assert universe.residues.n_residues == 5
    assert universe.atoms.n_atoms == count_atoms(root)
    assert len(universe.bonds) == count_bonds(root)


def _uneven_system() -> RootPrimitive:
    """Residue AAA has atoms at two depths: two carbons plus a nested group of two."""
    residue_a = CompositePrimitive(label="AAA")
    for i in range(2):
        residue_a.attach_child(AtomicPrimitive(ELEMENTS[6], label=f"C{i}"))
    tail = CompositePrimitive(label="tail")
    for i in range(2):
        tail.attach_child(AtomicPrimitive(ELEMENTS[6], label=f"T{i}"))
    residue_a.attach_child(tail)

    residue_b = CompositePrimitive(label="BBB")
    for i in range(2):
        residue_b.attach_child(AtomicPrimitive(ELEMENTS[6], label=f"C{i}"))

    chain = CompositePrimitive(label="chain", children=[residue_a, residue_b])
    return tag_roles(RootPrimitive(children=[chain]))


def test_uneven_hierarchy_keeps_residues_contiguous():
    """
    Atoms at different depths come out of the atom cross-section in
    breadth-first order, which would split residue AAA around residue BBB;
    the exporter must still keep each residue's atoms together.
    """
    universe = primitive_to_mdanalysis(_uneven_system(), {})
    assert list(universe.residues.resnames) == ["AAA", "BBB"]
    assert [residue.atoms.n_atoms for residue in universe.residues] == [4, 2]
    assert np.all(np.diff(universe.atoms.resindices) >= 0)


def test_no_segment_roles_raises(polyethylene_lexicon):
    """A hierarchy without roles cannot be exported."""
    chain = build_chain(polyethylene_lexicon, polyethylene_sequence(2))
    with pytest.raises(ValueError, match="No Primitive has role 'segment'"):
        primitive_to_mdanalysis(RootPrimitive(children=[chain]), {})


def test_segment_without_residue_roles_raises(polyethylene_lexicon):
    """A segment must contain residue-role Primitives."""
    chain = build_chain(polyethylene_lexicon, polyethylene_sequence(2))
    chain.role = "segment"
    with pytest.raises(ValueError, match="no Primitive with role 'residue'"):
        primitive_to_mdanalysis(RootPrimitive(children=[chain]), {})


def test_atoms_outside_any_residue_raise(polyethylene_lexicon):
    """Every atom must sit inside a residue-role Primitive."""
    chain = build_chain(polyethylene_lexicon, polyethylene_sequence(3))
    root = tag_roles(RootPrimitive(children=[chain]))
    next(iter(chain.children)).role = None  # untag the head unit
    with pytest.raises(ValueError, match="not inside both"):
        primitive_to_mdanalysis(root, {"ethane": "EAN", "tail": "TYL"})


def test_nested_residues_raise(polyethylene_lexicon):
    """A residue inside another residue is rejected."""
    chain = build_chain(polyethylene_lexicon, polyethylene_sequence(2))
    molecule = CompositePrimitive(label="mol", children=[chain])
    root = RootPrimitive(children=[molecule])
    molecule.role = "segment"
    chain.role = "residue"  # the chain and its units are both tagged as residues
    for unit in chain.children:
        unit.role = "residue"
    with pytest.raises(ValueError, match="contains another 'residue'-role"):
        primitive_to_mdanalysis(root, {})


def test_empty_root_raises():
    """A root with no children has no segments to export."""
    with pytest.raises(ValueError, match="No Primitive has role 'segment'"):
        primitive_to_mdanalysis(RootPrimitive(), {})


# ============================================================================
# POSITIONS
# ============================================================================


def test_positions_are_atom_centroids():
    """Atom positions are copied from the atoms' shapes."""
    toluene = primitive_from_smiles("Cc1ccccc1", label="toluene", embed_positions=True)
    universe = primitive_to_mdanalysis(single_molecule(toluene), {"toluene": "TOL"})

    expected = np.array([atom.shape.centroid for atom in toluene.children])
    assert np.allclose(universe.atoms.positions, expected, atol=1e-5)


def test_unpositioned_atoms_use_default_position(
    polyethylene_2mer, polyethylene_resname_map
):
    """Atoms without positions are placed at the default position."""
    strategy = AllAtomExportStrategy(default_atom_position=np.array([1.0, 2.0, 3.0]))
    universe = primitive_to_mdanalysis(
        polyethylene_2mer, polyethylene_resname_map, strategy
    )
    assert np.allclose(universe.atoms.positions, [1.0, 2.0, 3.0])


def test_invalid_default_position_raises():
    """The default position must be a 3-vector."""
    with pytest.raises(ValueError, match="3-dimensional"):
        AllAtomExportStrategy(default_atom_position=np.zeros(2))


# ============================================================================
# COARSE-GRAINED EXPORT (1 bead per repeat unit)
# ============================================================================


def test_cg_from_all_atom_hierarchy(polyethylene_multichain, polyethylene_resname_map):
    """Root -> chain -> repeat unit -> atom, exported as one bead per repeat unit."""
    universe = primitive_to_mdanalysis(
        polyethylene_multichain,
        polyethylene_resname_map,
        strategy=CoarseGrainedExportStrategy(),
    )
    n_units = sum(
        len(list(chain.children)) for chain in polyethylene_multichain.children
    )
    assert universe.atoms.n_atoms == n_units
    assert universe.residues.n_residues == n_units
    assert universe.segments.n_segments == 10
    assert len(universe.bonds) == n_units - 10  # linear chains
    for bond in universe.bonds:  # beads only bond within their own chain
        assert bond[0].segindex == bond[1].segindex
    assert universe.atoms.total_mass() == pytest.approx(
        sum(atom.mass for atom in polyethylene_multichain.cross_section(is_atom).nodes)
    )


def test_cg_bead_names_and_masses(polyethylene_3mer, polyethylene_resname_map):
    """Beads are named by residue name and weigh as much as their atoms."""
    universe = primitive_to_mdanalysis(
        polyethylene_3mer,
        polyethylene_resname_map,
        strategy=CoarseGrainedExportStrategy(),
    )
    assert list(universe.atoms.names) == ["HEA", "EAN", "TYL"]
    assert list(universe.atoms.types) == ["HEA", "EAN", "TYL"]
    chain = next(iter(polyethylene_3mer.children))
    assert np.allclose(universe.atoms.masses, [unit.mass for unit in chain.children])


def test_cg_bead_position_is_center_of_mass():
    """Beads sit at the mass-weighted center of their atoms."""
    toluene = primitive_from_smiles("Cc1ccccc1", label="toluene", embed_positions=True)
    universe = primitive_to_mdanalysis(
        single_molecule(toluene),
        {"toluene": "TOL"},
        strategy=CoarseGrainedExportStrategy(),
    )
    atoms = list(toluene.children)
    masses = np.array([atom.mass for atom in atoms])
    positions = np.array([atom.shape.centroid for atom in atoms])
    expected = (masses[:, None] * positions).sum(axis=0) / masses.sum()
    assert np.allclose(universe.atoms.positions[0], expected, atol=1e-5)


def _bead(label: str, ends: list[str]) -> SimplePrimitive:
    """A CG bead with a Connector for each chain end it links to."""
    bead = SimplePrimitive(label=label)
    if "in" in ends:
        bead.add_connector(
            Connector(
                anchor=AttachmentPoint({2}),
                linker=AttachmentPoint({1}),
                bondtype=BondType.SINGLE,
            )
        )
    if "out" in ends:
        bead.add_connector(
            Connector(
                anchor=AttachmentPoint({1}),
                linker=AttachmentPoint({2}),
                bondtype=BondType.SINGLE,
            )
        )
    return bead


def test_cg_from_bead_only_hierarchy():
    """Root -> chain -> repeat-unit beads, with no atoms anywhere."""
    lexicon = {
        "head": _bead("head", ["out"]),
        "mid": _bead("mid", ["in", "out"]),
        "tail": _bead("tail", ["in"]),
    }
    lengths = (5, 3)
    chains = []
    for n in lengths:
        chain = CompositePrimitive(label="cg_chain")
        chain.populate_from_topology_and_lexicon(
            path_graph_from_sequence(["head"] + (n - 2) * ["mid"] + ["tail"]),
            lexicon,
            set_label_from_lexicon=True,
        )
        chains.append(chain)
    root = tag_roles(RootPrimitive(children=chains))

    universe = primitive_to_mdanalysis(
        root,
        {"head": "HED", "mid": "MID", "tail": "TAL"},
        strategy=CoarseGrainedExportStrategy(),
    )
    assert universe.atoms.n_atoms == sum(lengths)
    assert universe.segments.n_segments == len(lengths)
    assert universe.bonds.indices.tolist() == [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
    ]
    assert list(universe.atoms.names) == [
        "HED",
        "MID",
        "MID",
        "MID",
        "TAL",
        "HED",
        "MID",
        "TAL",
    ]
    assert np.allclose(universe.atoms.positions, 0.0)  # beads have no positions
