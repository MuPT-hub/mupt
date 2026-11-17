"""
Faces: named groups of ports on a repeat unit

A port is a free Connector, identified by its label (the atom map number of its "*"
linker, else of its anchor atom). A face is a named, ordered tuple of port labels which
all bond to the SAME neighboring unit; its order is the strand order used when pairing
two faces at a junction.

Examples:
    linear unit  : {'head': (1,), 'tail': (2,)}           two width-1 faces
    ladder unit  : {'head': (1, 2), 'tail': (3, 4)}       two width-2 faces
    branched AB2 : {'a': (1,), 'b1': (2,), 'b2': (3,)}    three width-1 faces
"""

from types import MappingProxyType
from typing import Collection, Hashable, Mapping, Optional

from rdkit.Chem import Mol

from ...chemistry.rdkit.linkers import anchor_and_linker_idxs
from ...mupr.connection import Connector
from ...mupr.primitives import Primitive

from .._common import Face, FaceName, FaceSpec, PortLabel
from ..exceptions import FaceError


# Faces of a unit given without an explicit spec: the 1/2 anchor-map convention used by
# existing linear lexicons, e.g. "*-[CH2:1]-[CH2:2]-*" (read-only; copied by normalize_faces)
LINEAR_FACES: FaceSpec = MappingProxyType({"head": (1,), "tail": (2,)})


# Building and checking face specs
def _as_face(ports) -> Face:
    """Coerce a single port label or a sequence of labels into a Face tuple"""
    if isinstance(ports, (list, tuple)):
        return tuple(ports)
    return (ports,)  # a lone label (including a string) is a width-1 face


def normalize_faces(raw: Optional[Mapping[FaceName, object]]) -> dict[FaceName, Face]:
    """
    Turn a user-written face spec into {face name: tuple of port labels}

    None gives the linear 1/2 convention; a scalar value becomes a width-1 face
    """
    if raw is None:
        return dict(LINEAR_FACES)
    if not isinstance(raw, Mapping):
        raise FaceError(f"Faces must be given as a mapping of face name to ports, got {raw!r}")

    faces = {name: _as_face(ports) for name, ports in raw.items()}

    # each face needs at least one port, and no port may appear twice anywhere
    seen: dict[PortLabel, FaceName] = {}
    for name, face in faces.items():
        if not face:
            raise FaceError(f"Face {name!r} has no ports")
        for port in face:
            if port in seen:
                raise FaceError(
                    f"Port {port!r} appears in face {seen[port]!r} and again in face {name!r}"
                )
            seen[port] = name

    return faces


# Looking up ports on Primitives
def free_ports_by_label(prim: Primitive) -> dict[PortLabel, Connector]:
    """
    Map each port label to its free Connector on a Primitive

    Only free Connectors are searched: internal Connectors carry map-number labels too,
    so they would give false matches. Labels must be unique among the free Connectors.
    """
    ports: dict[PortLabel, Connector] = {}
    for conn in prim.connections.connectors_free:
        if conn.label in ports:
            raise FaceError(f"{prim.label!r} has more than one free port labelled {conn.label!r}")
        ports[conn.label] = conn

    return ports


def validate_faces(
    prim: Primitive,
    faces: FaceSpec,
    *,
    required: Collection[FaceName] = (),
    label: Optional[Hashable] = None,
) -> None:
    """
    Check that a face spec describes a unit's free ports exactly

    Every listed port must be a free port, every free port must belong to a face,
    every required face must exist, and all ports of a face must share one bond type
    """
    name = prim.label if (label is None) else label  # used only in error messages
    ports = free_ports_by_label(prim)

    if missing_faces := set(required) - set(faces):
        raise FaceError(f"Unit {name!r} is missing required face(s) {sorted(map(str, missing_faces))}")

    listed = {port for face in faces.values() for port in face}
    if unknown := listed - set(ports):
        raise FaceError(
            f"Faces of unit {name!r} list port(s) {sorted(map(str, unknown))}, "
            f"which aren't free ports (free ports: {sorted(map(str, ports))})"
        )
    if unlisted := set(ports) - listed:
        raise FaceError(f"Free port(s) {sorted(map(str, unlisted))} of unit {name!r} aren't in any face")

    # bonds within one face go to the same neighbor, so they must all be the same kind
    for face_name, face in faces.items():
        bondtypes = {ports[port].bondtype for port in face}
        if len(bondtypes) > 1:
            raise FaceError(f"Ports of face {face_name!r} on unit {name!r} have mixed bond types {bondtypes}")


def strip_port_map_numbers(prim: Primitive, faces: FaceSpec) -> None:
    """
    Remove atom map numbers from the anchor atoms of a unit's ports

    Otherwise exporters would write them back onto real atoms; map numbers on
    non-port atoms are left alone, since they may be user annotations
    """
    ports = free_ports_by_label(prim)
    for face in faces.values():
        for port in face:
            ports[port].holder.metadata.pop("molAtomMapNumber", None)


# RDKit-level helpers (for stereo and symmetry analysis, before import)
def port_atom_indices(mol: Mol) -> dict[PortLabel, tuple[int, int]]:
    """Map port labels to (anchor, linker) atom indices, labelling ports as the importer does"""
    ports: dict[PortLabel, tuple[int, int]] = {}
    for anchor_idx, linker_idx in anchor_and_linker_idxs(mol):
        # same precedence as the importer's labeller: "*" map number, then anchor's, then default
        label = (
            mol.GetAtomWithIdx(linker_idx).GetAtomMapNum()
            or mol.GetAtomWithIdx(anchor_idx).GetAtomMapNum()
            or Connector.DEFAULT_LABEL
        )
        if label in ports:
            raise FaceError(f"More than one port labelled {label!r}")
        ports[label] = (anchor_idx, linker_idx)

    return ports


def face_anchor_indices(mol: Mol, faces: FaceSpec, name: FaceName) -> tuple[int, ...]:
    """Anchor atom indices of one face's ports, in strand order"""
    if name not in faces:
        raise FaceError(f"No face named {name!r} (faces: {list(faces)})")

    ports = port_atom_indices(mol)
    if missing := [port for port in faces[name] if port not in ports]:
        raise FaceError(f"Face {name!r} lists port(s) {missing} not found in the molecule")

    return tuple(ports[port][0] for port in faces[name])
