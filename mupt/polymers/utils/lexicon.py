"""
Lexicons: building the repeat-unit templates that descriptive graphs refer to

A lexicon maps unit keys to template Primitives. Generated graphs use keys of the form
(label, config): config is +1 / -1 for the two stereo variants of a tactic unit and None
otherwise. Every template is ready to copy as is: its faces are checked, the map numbers
used as port labels are stripped from its atoms, and residue metadata is set on it, so
each realized copy inherits that metadata.
"""

from typing import Collection, Literal, Mapping, Optional

from rdkit.Chem import Mol
from rdkit.Chem.rdmolfiles import MolFromSmiles

from ...chemistry.rdkit.linkers import embed_with_linker_placeholders
from ...chemistry.rdkit.sanitization import sanitized_mol
from ...chemistry.smiles import DEFAULT_SMILES_READ_PARAMS
from ...interfaces.rdkit import primitive_from_rdkit, rdkit_mol_from_primitive
from ...mupr.primitives import Primitive

from .._common import CONFIG_KEY, RESIDUE_NAME_KEY, UNIT_LABEL_KEY, Config, FaceSpec, GeneratedKey, UnitLabel
from ..exceptions import FaceError
from ..rules.faces import free_ports_by_label, normalize_faces, strip_port_map_numbers, validate_faces
from .stereo import backbone_stereocenters, stereo_variant, strand_paths

type UnitLike = str | Primitive  # a repeat unit, as SMILES or as a Primitive
type CapSide = Literal["head", "tail"]

# Cap templates use reserved labels in their lexicon keys, so they can't collide with
# user unit labels; their residue metadata uses the plainer names below
CAP_KEY_LABELS: dict[CapSide, str] = {"head": "<head_cap>", "tail": "<tail_cap>"}
CAP_RESIDUE_NAMES: dict[CapSide, str] = {"head": "head_cap", "tail": "tail_cap"}
CAP_FACE = "cap"


# RDKit round trip
def unit_mol(unit: UnitLike) -> Mol:
    """H-explicit RDKit Mol for a unit given as SMILES or as a Primitive"""
    if isinstance(unit, str):
        return sanitized_mol(MolFromSmiles(unit, params=DEFAULT_SMILES_READ_PARAMS), add_Hs=True)

    # exported ports keep their labels as map numbers on the "*" atoms
    return rdkit_mol_from_primitive(unit)


def embedded_primitive(mol: Mol, label: UnitLabel, embed_random_seed: int = 0) -> Primitive:
    """Embed coordinates for a Mol (respecting its chiral tags) and import it as a Primitive"""
    mol = Mol(mol)  # embedding adds a conformer; don't modify the caller's Mol
    conformer_idx = embed_with_linker_placeholders(mol, random_seed=embed_random_seed)
    if conformer_idx < 0:
        raise ValueError(f"RDKit failed to embed coordinates for unit {label!r}")

    return primitive_from_rdkit(mol, conformer_idx=conformer_idx, label=label)


# Repeat units
def build_unit_templates(
    label: UnitLabel,
    unit: UnitLike,
    faces: FaceSpec,
    *,
    tactic: bool = False,
    side_group: Optional[int] = None,
    required_faces: Collection[str] = (),
    embed_random_seed: int = 0,
) -> dict[GeneratedKey, Primitive]:
    """
    Build the lexicon entries for one repeat unit

    Returns {(label, 1): ..., (label, -1): ...} when tactic is set and the unit has
    stereocenters on its backbone (between its 'head' and 'tail' faces), and
    {(label, None): ...} otherwise. A Primitive with no stereo variants to build is
    copied as is, keeping its geometry; everything else is embedded through RDKit.

    side_group is the atom map number of the atom to use as a stereocenter's side
    group X when defining configs (see utils.stereo); by default the side neighbor
    of highest CIP rank is used. Its map number is stripped from the templates.
    """
    variants: dict[Optional[Config], Primitive] = {}

    if tactic:
        # stereo is set at the RDKit level, before embedding, relative to the backbone
        mol = unit_mol(unit)
        paths = _backbone_paths(mol, faces, label)
        centers = backbone_stereocenters(mol, paths)
        side_groups = _side_group_atoms(mol, side_group, faces, label)
        if centers:
            for config in (1, -1):
                variant = stereo_variant(mol, centers, config, paths, side_groups)
                variants[config] = embedded_primitive(variant, label, embed_random_seed)

    if not variants:  # not tactic, or no backbone stereocenters to vary
        if isinstance(unit, Primitive):
            variants[None] = unit.copy()  # keep user-provided geometry
        else:
            variants[None] = embedded_primitive(unit_mol(unit), label, embed_random_seed)

    # finish every template the same way, so all copies are ready to bond
    for config, template in variants.items():
        _finish_template(template, faces, required_faces, label)
        if side_group is not None:
            _strip_atom_map_number(template, side_group)  # an annotation for us, not for exports
        template.metadata.update({
            UNIT_LABEL_KEY: label,
            CONFIG_KEY: config,
            RESIDUE_NAME_KEY: str(label),
        })

    return {(label, config): template for config, template in variants.items()}


def _backbone_paths(mol: Mol, faces: FaceSpec, label: UnitLabel) -> list[list[int]]:
    """A tactic unit's backbone strands, which run from its 'head' face to its 'tail' face"""
    if not {"head", "tail"} <= set(faces):
        raise FaceError(f"Tactic unit {label!r} needs 'head' and 'tail' faces to define its backbone")

    return strand_paths(mol, faces)


def _side_group_atoms(mol: Mol, side_group: Optional[int], faces: FaceSpec, label: UnitLabel) -> list[int]:
    """Atom indices of the user-chosen side group X, found by its atom map number"""
    if side_group is None:
        return []
    if any(side_group in face for face in faces.values()):
        raise FaceError(f"Side group map number {side_group} of unit {label!r} is also a port label")

    atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == side_group]
    if not atoms:
        raise FaceError(f"Unit {label!r} has no atom with map number {side_group} to use as its side group")
    return atoms


def _strip_atom_map_number(template: Primitive, map_number: int) -> None:
    """Remove one atom map number from a template's atoms"""
    for atom in template.leaves:
        if atom.metadata.get("molAtomMapNumber") == map_number:
            del atom.metadata["molAtomMapNumber"]


def _finish_template(template: Primitive, faces: FaceSpec, required_faces: Collection[str], label) -> None:
    """Check a template's faces, then strip the map numbers used as port labels"""
    validate_faces(template, faces, required=required_faces, label=label)
    strip_port_map_numbers(template, faces)  # port labels are already set, so lookup still works


# End groups
def cap_key(side: CapSide) -> GeneratedKey:
    """Lexicon key of the cap template for one chain end"""
    return (CAP_KEY_LABELS[side], None)


def build_cap_template(
    cap: UnitLike,
    side: CapSide,
    *,
    label: Optional[str] = None,
    faces: Optional[Mapping] = None,
    embed_random_seed: int = 0,
) -> tuple[Primitive, FaceSpec]:
    """
    Build the template for a chain-end cap, along with its faces

    Without faces, a cap has a single face, 'cap', holding all of its free ports in
    ascending label order. Given faces must include 'cap' (the ports that bond the
    chain); any other face stays free once the cap is attached.

    An anonymous cap (label None) is keyed cap_key(side); a named one is keyed
    (label, None) and uses its label as its residue name. Returns (template, faces).
    """
    if side not in CAP_KEY_LABELS:
        raise ValueError(f"Cap side must be 'head' or 'tail', got {side!r}")
    name = CAP_RESIDUE_NAMES[side] if (label is None) else label

    if isinstance(cap, Primitive):
        template = cap.copy()
    else:
        template = embedded_primitive(unit_mol(cap), name, embed_random_seed)

    if faces is None:
        ports = _sorted_labels(free_ports_by_label(template))
        if not ports:
            raise FaceError(f"The {side} cap has no free ports to bond to the chain")
        cap_faces = {CAP_FACE: ports}
    else:
        cap_faces = normalize_faces(faces)
        if CAP_FACE not in cap_faces:
            raise FaceError(f"The faces of cap {name!r} need a {CAP_FACE!r} face (the ports that bond the chain)")

    _finish_template(template, cap_faces, (CAP_FACE,), name)
    template.metadata.update({
        UNIT_LABEL_KEY: name,
        CONFIG_KEY: None,
        RESIDUE_NAME_KEY: name,
    })

    return template, cap_faces


def _sorted_labels(labels) -> tuple:
    """Port labels in ascending order (by text, if the labels can't be compared directly)"""
    try:
        return tuple(sorted(labels))
    except TypeError:
        return tuple(sorted(labels, key=str))
