'''Readers which convert RDKit Atoms and Mols into the MuPT molecular representation'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'

from typing import (
    Hashable,
    Optional,
)
import json
from pathlib import Path
from collections.abc import Sequence

from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit.Chem.rdchem import (
    Atom,
    Mol,
)
from rdkit.Chem.rdmolops import GetMolFrags

from ...chemistry.linkers import is_linker
from .components import atom_positions_from_rdkit, connector_between_rdatoms

from .labelling import name_for_rdkit_mol
from ...geometry.shapes import PointCloud
from ...chemistry.smiles import DEFAULT_SMILES_WRITE_PARAMS, SmilesWriteParams
from ...chemistry.conversion import rdkit_atom_to_element

from ...mupr.primitives import Primitive, PrimitiveHandle
from ...mupr.connection import TraversalDirection
from ...roles import PrimitiveRole

from .exporters import (
    MUPT_HIERARCHY_PATH,
    MUPT_ROOT_METADATA_COUNT,
    MUPT_ROOT_METADATA_KEY_PREFIX,
    MUPT_ROOT_METADATA_VALUE_PREFIX,
    MUPT_SAAMR_SDF_KIND,
    MUPT_SAAMR_SDF_VERSION,
    MUPT_SERIALIZATION_KIND,
    MUPT_SERIALIZATION_VERSION,
)


def primitive_from_rdkit_atom(
    parent_mol : Mol,
    atom_idx : int,
    conformer_idx : Optional[int]=None,
    attach_connectors : bool=False,
    role : PrimitiveRole=PrimitiveRole.PARTICLE,
    **kwargs
) -> Primitive:
    '''Initialize an atomic Primitive from an RDKit Atom'''
    atom : Atom = parent_mol.GetAtomWithIdx(atom_idx)
    atom_primitive = Primitive(
        element=rdkit_atom_to_element(atom),
        label=atom_idx,
        metadata=atom.GetPropsAsDict(
            includePrivate=True,
            includeComputed=False, # NOTE: computed props suppressed to avoid "unpicklable RDKit vector" errors 
        ), 
        role=role,
    )
    if (map_num := atom.GetAtomMapNum()) != 0:
        atom_primitive.metadata['molAtomMapNumber'] = map_num
    
    atom_pos = atom_positions_from_rdkit(parent_mol, conformer_idx=conformer_idx, atom_idxs=[atom_idx])
    if atom_pos is not None:
        atom_primitive.shape = PointCloud(positions=atom_pos[0, :]) # extract as vector from 2D array
    
    if attach_connectors:
        for nb_atom in atom.GetNeighbors(): # TODO: decide how bond Props should be split among metadata of the two bonded atoms
            atom_primitive.register_connector(
                connector_between_rdatoms(
                    parent_mol=parent_mol,
                    from_atom_idx=atom_idx,
                    to_atom_idx=nb_atom.GetIdx(),
                    conformer_idx=conformer_idx,
                    **kwargs,
                )
            )
    return atom_primitive
    
def primitive_from_rdkit_chain(
    rdmol_chain : Mol,
    conformer_idx : Optional[int]=None,
    label : Optional[Hashable]=None,
    role : PrimitiveRole=PrimitiveRole.RESIDUE,
    atom_role : PrimitiveRole=PrimitiveRole.PARTICLE,
    atom_label : str='ATOM',
    external_linker_label : str='*',
    smiles_writer_params : SmilesWriteParams=DEFAULT_SMILES_WRITE_PARAMS,
    **kwargs,
) -> Primitive:
    ''' 
    Initialize a Primitive hierarchy from an RDKit Mol representing a single molecule

    Parameters
    ----------
    rdmol : Chem.Mol
        The RDKit Mol object to convert
    conformer_idx : int, optional
        The ID of the conformer to use, by default None (uses no conformer)
    label : Hashable, optional
        A distinguishing label for the Primitive
        If none is provided, the canonicalized SMILES of the RDKit Mol will be used
    
    Returns
    -------
    Primitive
        The created Primitive object
    '''
    if label is None:
        label = name_for_rdkit_mol(rdmol_chain, smiles_writer_params=smiles_writer_params)
    rdmol_primitive = Primitive(
        label=label,
        metadata=rdmol_chain.GetPropsAsDict(includePrivate=True, includeComputed=False),
        role=role,
    )
    ## DEV: opting to not inject stereochemical metadata for now, since that may change as Primitive repr is transformed geometrically
    # stereo_info_map : dict[int, StereoInfo] = {
    #     stereo_info.centeredOn : stereo_info # TODO: determine most appropriate choice of flags to use in FindPotentialStereo
    #         for stereo_info in FindPotentialStereo(rdmol_chain, cleanIt=True, flagPossible=True) 
    # } 

    # 1) Insert child Primitives for each atom (EVEN linkers - this keeps indices in sync for final handle assignment)
    linker_idxs : set[int] = set()
    atom_idx_to_handle_map : dict[int, PrimitiveHandle] = dict() # DEV: as-implemented, handle idx **SHOULD** match atom idx, but it never hurts to be explicit :P
    for atom in rdmol_chain.GetAtoms(): # DEV: opting not to get atoms implicitly from bonds to handle single, unbonded atom (e.g. noble gas) uniformly
        atom_idx = atom.GetIdx()
        if is_linker(atom):
            linker_idxs.add(atom_idx)
        
        atom_prim = primitive_from_rdkit_atom(
            rdmol_chain,
            atom_idx,
            conformer_idx=conformer_idx,
            attach_connectors=False, # will attach per-bond to avoid needing to match connector handles to bond idxs
            role=atom_role,
        )
        atom_idx_to_handle_map[atom_idx] = rdmol_primitive.attach_child(atom_prim, label=atom_label)
    
    # 2) forge connections between Primitives corresponding to bonded atoms (propagating external Connectors up to mol primitive)
    for bond in rdmol_chain.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        ## Primitive 1 + associated Connector
        begin_prim_handle = atom_idx_to_handle_map[begin_idx]
        begin_prim = rdmol_primitive.fetch_child(begin_prim_handle)
        begin_conn = connector_between_rdatoms(
            rdmol_chain,
            from_atom_idx=begin_idx,
            to_atom_idx=end_idx,
            conformer_idx=conformer_idx,
            **kwargs,
        )
        begin_conn_handle = begin_prim.register_connector(begin_conn)
        rdmol_primitive.bind_external_connector(begin_prim_handle, begin_conn_handle, label=external_linker_label)
        
        ### Primitive 2 + associated Connector
        end_prim_handle = atom_idx_to_handle_map[end_idx]
        end_prim = rdmol_primitive.fetch_child(end_prim_handle)
        end_conn = connector_between_rdatoms(
            rdmol_chain,
            from_atom_idx=end_idx,
            to_atom_idx=begin_idx,
            conformer_idx=conformer_idx,
            **kwargs,
        )
        end_conn_handle = end_prim.register_connector(end_conn)
        rdmol_primitive.bind_external_connector(end_prim_handle, end_conn_handle, label=external_linker_label)
        
        ### joining of the pair of Connectors
        rdmol_primitive.connect_children(
            begin_prim_handle,
            begin_conn_handle,
            end_prim_handle,
            end_conn_handle,
        )

    # 3) excise temporary linker Primitives no longer needed as doorstops
    for linker_idx in linker_idxs:
        rdmol_primitive.detach_child(atom_idx_to_handle_map[linker_idx])

    ## 3a) insert traversal direction info based on 1-2 map number convention
    for ext_conn_handle, conn_ref in rdmol_primitive.external_connectors.items():
        atom_primitive = rdmol_primitive.fetch_child(conn_ref.primitive_handle)
        ext_conn = rdmol_primitive.fetch_connector(ext_conn_handle)
        
        if (mapnum := atom_primitive.metadata.get('molAtomMapNumber')) in {1,2}:
            chain_direction = TraversalDirection(mapnum)
            ext_conn.anchor.attachables.add(chain_direction)
            ext_conn.linker.attachables.add(TraversalDirection.complement(chain_direction))

    # 4) Inject conformer info - DEV: there are many avenues to do this (e.g. collate shape from children, if not None on all), but opted for the simplest for now
    non_linker_conformer = atom_positions_from_rdkit(
        rdmol_chain,
        conformer_idx=conformer_idx,
        atom_idxs=sorted(atom_idx_to_handle_map.keys() - linker_idxs), # preserve atom order
    )
    if non_linker_conformer is not None: # can't just check if Falsy in case this is an array (would need all() then)
        rdmol_primitive.shape = PointCloud(positions=non_linker_conformer) # exploit default NoneType value
    rdmol_primitive.check_self_consistent()
        
    return rdmol_primitive


def _atom_string_prop(atom: Atom, key: str, default: str) -> str:
    """Fetch a string atom property with a fallback value."""
    return atom.GetProp(key) if atom.HasProp(key) else default


def _atom_int_prop(atom: Atom, key: str, default: int) -> int:
    """Fetch an integer atom property with a fallback value."""
    return atom.GetIntProp(key) if atom.HasProp(key) else default


def _residue_key(atom: Atom) -> tuple[str, str, int, str]:
    """Return the residue identity key for an RDKit atom."""
    # Prefer MuPT export metadata, falling back to PDB residue info when present.
    pdb_info = atom.GetPDBResidueInfo()
    chain_id = pdb_info.GetChainId() if pdb_info is not None else ""
    insertion_code = pdb_info.GetInsertionCode() if pdb_info is not None else ""
    resid = pdb_info.GetResidueNumber() if pdb_info is not None else 1
    resname = pdb_info.GetResidueName().strip() if pdb_info is not None else "RES"
    return (
        "mupt" if atom.HasProp("mupt_residue_index") else chain_id,
        insertion_code,
        _atom_int_prop(atom, "mupt_residue_index", resid),
        _atom_string_prop(atom, "mupt_residue_label", resname),
    )


def _non_root_metadata(rdmol: Mol) -> dict:
    """Return Mol properties excluding namespaced UNIVERSE-root metadata."""
    root_metadata_keys = {MUPT_ROOT_METADATA_COUNT}
    if rdmol.HasProp(MUPT_ROOT_METADATA_COUNT):
        for idx in range(rdmol.GetIntProp(MUPT_ROOT_METADATA_COUNT)):
            root_metadata_keys.add(f"{MUPT_ROOT_METADATA_KEY_PREFIX}{idx}")
            root_metadata_keys.add(f"{MUPT_ROOT_METADATA_VALUE_PREFIX}{idx}")

    return {
        key: value
        for key, value in rdmol.GetPropsAsDict(includePrivate=True, includeComputed=False).items()
        if key not in root_metadata_keys
    }


def _root_metadata(rdmol: Mol) -> dict:
    """Return namespaced UNIVERSE-root metadata from an exported RDKit Mol."""
    if not rdmol.HasProp(MUPT_ROOT_METADATA_COUNT):
        return {}
    props = rdmol.GetPropsAsDict(includePrivate=True, includeComputed=False)
    return {
        rdmol.GetProp(f"{MUPT_ROOT_METADATA_KEY_PREFIX}{idx}"): props[
            f"{MUPT_ROOT_METADATA_VALUE_PREFIX}{idx}"
        ]
        for idx in range(rdmol.GetIntProp(MUPT_ROOT_METADATA_COUNT))
    }


def _fast_segment_metadata(rdmol: Mol) -> dict:
    """Return segment metadata without MuPT serialization transport keys."""
    return {
        key: value
        for key, value in _non_root_metadata(rdmol).items()
        if key not in {MUPT_SERIALIZATION_KIND, MUPT_SERIALIZATION_VERSION}
    }


def _apply_traversal_direction(connector, atom_primitive: Primitive) -> None:
    """Annotate linker connectors from RDKit map-number direction markers."""
    if (mapnum := atom_primitive.metadata.get('molAtomMapNumber')) in {1, 2}:
        chain_direction = TraversalDirection(mapnum)
        connector.anchor.attachables.add(chain_direction)
        connector.linker.attachables.add(TraversalDirection.complement(chain_direction))


def _child_handle(parent: Primitive, child: Primitive) -> PrimitiveHandle:
    """Return the parent-local handle for a known child Primitive."""
    for handle, candidate in parent.children_by_handle.items():
        if candidate is child:
            return handle
    raise ValueError(f"Child '{child.label}' is not attached to parent '{parent.label}'")


def _primitive_role_from_path_entry(entry: dict) -> PrimitiveRole:
    """Convert a serialized path role string into a PrimitiveRole."""
    try:
        return PrimitiveRole(entry["role"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid MuPT hierarchy path role entry: {entry}") from exc


def _path_key(path_entries: list[dict], stop: int) -> tuple[tuple[str, int, str], ...]:
    """Return a stable key for a serialized path prefix."""
    return tuple(
        (
            str(entry["handle_label"]),
            int(entry["handle_index"]),
            str(entry["role"]),
        )
        for entry in path_entries[:stop]
    )


def _path_entry_identity(entry: dict) -> tuple[str, str, str, int]:
    """Return the structural identity fields for one serialized path entry."""
    return (
        str(entry["role"]),
        str(entry["label"]),
        str(entry["handle_label"]),
        int(entry["handle_index"]),
    )


def _validate_path_entry_match(reference: dict, candidate: dict, atom_idx: int, depth: int) -> None:
    """Reject inconsistent repeated path prefixes within one MuPT SDF record."""
    if _path_entry_identity(candidate) != _path_entry_identity(reference):
        raise ValueError(
            "MuPT hierarchy path prefix is inconsistent for "
            f"atom {atom_idx} at depth {depth}."
        )


def _validate_mupt_saamr_mol(rdmol: Mol) -> None:
    """Validate MuPT SAAMR SDF serialization markers on one RDKit Mol."""
    if not rdmol.HasProp(MUPT_SERIALIZATION_KIND) or not rdmol.HasProp(MUPT_SERIALIZATION_VERSION):
        raise ValueError(
            "RDKit Mol is missing MuPT SAAMR SDF serialization metadata. "
            "Use primitive_to_rdkit_mols() to create MuPT-generated SDF input."
        )
    if rdmol.GetProp(MUPT_SERIALIZATION_KIND) != MUPT_SAAMR_SDF_KIND:
        raise ValueError(f"Unsupported MuPT serialization kind: {rdmol.GetProp(MUPT_SERIALIZATION_KIND)}")
    if rdmol.GetProp(MUPT_SERIALIZATION_VERSION) != MUPT_SAAMR_SDF_VERSION:
        raise ValueError(
            f"Unsupported MuPT SAAMR SDF version: {rdmol.GetProp(MUPT_SERIALIZATION_VERSION)}"
        )


def _load_mupt_sdf_mols(source: str | Path | Sequence[str | Path]) -> list[Mol]:
    """Load one or more RDKit Mols from MuPT SDF path input."""
    if isinstance(source, (str, Path)):
        sources = [source]
    else:
        sources = list(source)

    mols: list[Mol] = []
    for sdf_path in sources:
        supplier = SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        for mol in supplier:
            if mol is not None:
                mols.append(mol)
    if not mols:
        raise ValueError("No RDKit molecules were loaded from MuPT SDF input")
    return mols


def _validate_root_metadata_consistency(rdmols: Sequence[Mol]) -> dict:
    """Return shared MuPT root metadata after verifying all records agree."""
    reference = _root_metadata(rdmols[0])
    for mol_idx, rdmol in enumerate(rdmols[1:], start=1):
        metadata = _root_metadata(rdmol)
        if metadata != reference:
            raise ValueError(
                "MuPT SDF records disagree on duplicated UNIVERSE metadata at "
                f"record {mol_idx}."
            )
    return reference


def _enclosing_role(node: Primitive, role: PrimitiveRole) -> Primitive:
    """Return the nearest ancestor-or-self carrying the requested role."""
    current = node
    while current is not None:
        if current.role == role:
            return current
        current = current.parent
    raise ValueError(f"Node '{node.label}' is not enclosed by role {role}")


def _bind_connector_up_to_owner(
    atom: Primitive,
    atom_handle: PrimitiveHandle,
    atom_parent: Primitive,
    connector,
    owner: Primitive,
    external_linker_label: str,
) -> tuple[PrimitiveHandle, object, list]:
    """Mirror an atom connector upward until it is external on the owner."""
    current_child = atom
    current_parent = atom_parent
    current_handle = atom_handle
    current_conn_handle = atom.register_connector(connector)
    mirrored_connectors = [atom.fetch_connector(current_conn_handle)]

    while current_parent is not owner:
        current_conn_handle = current_parent.bind_external_connector(
            current_handle,
            current_conn_handle,
            label=external_linker_label,
        )
        mirrored_connectors.append(current_parent.fetch_connector(current_conn_handle))
        current_child = current_parent
        current_parent = current_parent.parent
        current_handle = _child_handle(current_parent, current_child)

    owner.bind_external_connector(current_handle, current_conn_handle, label=external_linker_label)
    return current_handle, current_conn_handle, mirrored_connectors


def _mirror_linker_bond_to_owner(
    rdmol: Mol,
    atom_idx: int,
    linker_idx: int,
    atom_parent: Primitive,
    atom_handle: PrimitiveHandle,
    owner: Primitive,
    conformer_idx: Optional[int],
    external_linker_label: str,
    **kwargs,
) -> None:
    """Rebuild one atom-linker bond as external connectors up to the owner."""
    atom_primitive = atom_parent.fetch_child(atom_handle)
    _, _, mirrored_connectors = _bind_connector_up_to_owner(
        atom_primitive,
        atom_handle,
        atom_parent,
        connector_between_rdatoms(
            rdmol,
            from_atom_idx=atom_idx,
            to_atom_idx=linker_idx,
            conformer_idx=conformer_idx,
            **kwargs,
        ),
        owner,
        external_linker_label,
    )
    for mirrored_connector in mirrored_connectors:
        _apply_traversal_direction(mirrored_connector, atom_primitive)


def _connect_rdkit_atom_pair_at_owner(
    rdmol: Mol,
    begin_idx: int,
    end_idx: int,
    begin_parent: Primitive,
    begin_handle: PrimitiveHandle,
    end_parent: Primitive,
    end_handle: PrimitiveHandle,
    owner: Primitive,
    conformer_idx: Optional[int],
    external_linker_label: str,
    **kwargs,
) -> None:
    """Rebuild one RDKit atom-atom bond at its owning MuPT node."""
    begin_atom = begin_parent.fetch_child(begin_handle)
    end_atom = end_parent.fetch_child(end_handle)
    begin_child_handle, begin_conn_handle, _ = _bind_connector_up_to_owner(
        begin_atom,
        begin_handle,
        begin_parent,
        connector_between_rdatoms(
            rdmol,
            from_atom_idx=begin_idx,
            to_atom_idx=end_idx,
            conformer_idx=conformer_idx,
            **kwargs,
        ),
        owner,
        external_linker_label,
    )
    end_child_handle, end_conn_handle, _ = _bind_connector_up_to_owner(
        end_atom,
        end_handle,
        end_parent,
        connector_between_rdatoms(
            rdmol,
            from_atom_idx=end_idx,
            to_atom_idx=begin_idx,
            conformer_idx=conformer_idx,
            **kwargs,
        ),
        owner,
        external_linker_label,
    )
    owner.connect_children(
        begin_child_handle,
        begin_conn_handle,
        end_child_handle,
        end_conn_handle,
    )


def _rebuild_shape_from_leaf_positions(node: Primitive) -> None:
    """Assign a PointCloud shape when a container has positioned leaf atoms."""
    positions = [atom.shape.centroid for atom in node.leaves if atom.shape is not None]
    if positions:
        node.shape = PointCloud(positions=positions)


def _primitive_from_mupt_saamr_mol(
    rdmol_segment: Mol,
    conformer_idx: Optional[int],
    external_linker_label: str,
    reconstruct_bonds: bool,
    reconstruct_shapes: bool,
    **kwargs,
) -> Primitive:
    """Rebuild one SEGMENT tree from MuPT SAAMR SDF atom path metadata."""
    _validate_mupt_saamr_mol(rdmol_segment)

    segment: Primitive | None = None
    nodes_by_path: dict[tuple[tuple[str, int, str], ...], Primitive] = {}
    path_entries_by_key: dict[tuple[tuple[str, int, str], ...], dict] = {}
    atom_idx_to_parent: dict[int, Primitive] = {}
    atom_idx_to_handle: dict[int, PrimitiveHandle] = {}
    linker_idxs: set[int] = set()

    for atom in rdmol_segment.GetAtoms():
        atom_idx = atom.GetIdx()
        if is_linker(atom):
            linker_idxs.add(atom_idx)
            continue
        if not atom.HasProp(MUPT_HIERARCHY_PATH):
            raise ValueError(f"Atom {atom_idx} is missing {MUPT_HIERARCHY_PATH}")

        path_entries = json.loads(atom.GetProp(MUPT_HIERARCHY_PATH))
        if not path_entries:
            raise ValueError(f"Atom {atom_idx} has an empty MuPT hierarchy path")
        if _primitive_role_from_path_entry(path_entries[0]) != PrimitiveRole.SEGMENT:
            raise ValueError("MuPT SAAMR SDF atom paths must start at SEGMENT role")
        if _primitive_role_from_path_entry(path_entries[-1]) != PrimitiveRole.PARTICLE:
            raise ValueError("MuPT SAAMR SDF atom paths must end at PARTICLE role")

        if segment is None:
            segment = Primitive(
                label=str(path_entries[0]["label"]),
                metadata=_fast_segment_metadata(rdmol_segment),
                role=PrimitiveRole.SEGMENT,
            )
            segment_key = _path_key(path_entries, 1)
            nodes_by_path[segment_key] = segment
            path_entries_by_key[segment_key] = path_entries[0]
        else:
            _validate_path_entry_match(path_entries_by_key[_path_key(path_entries, 1)], path_entries[0], atom_idx, 1)

        parent = segment
        for depth in range(2, len(path_entries)):
            entry = path_entries[depth - 1]
            prefix_key = _path_key(path_entries, depth)
            if prefix_key not in nodes_by_path:
                metadata = {}
                if _primitive_role_from_path_entry(entry) == PrimitiveRole.RESIDUE:
                    pdb_info = atom.GetPDBResidueInfo()
                    insertion_code = pdb_info.GetInsertionCode() if pdb_info is not None else ""
                    if insertion_code:
                        metadata["pdb_insertion_code"] = insertion_code
                child = Primitive(
                    label=str(entry["label"]),
                    metadata=metadata,
                    role=_primitive_role_from_path_entry(entry),
                )
                parent.attach_child(child, label=str(entry["handle_label"]))
                nodes_by_path[prefix_key] = child
                path_entries_by_key[prefix_key] = entry
            else:
                _validate_path_entry_match(path_entries_by_key[prefix_key], entry, atom_idx, depth)
            parent = nodes_by_path[prefix_key]

        particle_entry = path_entries[-1]
        atom_primitive = primitive_from_rdkit_atom(
            rdmol_segment,
            atom_idx,
            conformer_idx=conformer_idx,
            attach_connectors=False,
            role=PrimitiveRole.PARTICLE,
        )
        atom_primitive.label = str(particle_entry["label"])
        atom_idx_to_parent[atom_idx] = parent
        atom_idx_to_handle[atom_idx] = parent.attach_child(
            atom_primitive,
            label=str(particle_entry["handle_label"]),
        )

    if segment is None:
        raise ValueError("MuPT SAAMR SDF segment contained no non-linker atoms")

    if reconstruct_bonds:
        for bond in rdmol_segment.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if begin_idx in linker_idxs or end_idx in linker_idxs:
                if begin_idx in linker_idxs and end_idx in linker_idxs:
                    continue
                atom_idx, linker_idx = (end_idx, begin_idx) if begin_idx in linker_idxs else (begin_idx, end_idx)
                _mirror_linker_bond_to_owner(
                    rdmol_segment,
                    atom_idx,
                    linker_idx,
                    atom_idx_to_parent[atom_idx],
                    atom_idx_to_handle[atom_idx],
                    segment,
                    conformer_idx,
                    external_linker_label,
                    **kwargs,
                )
                continue

            begin_parent = atom_idx_to_parent[begin_idx]
            end_parent = atom_idx_to_parent[end_idx]
            begin_atom = begin_parent.fetch_child(atom_idx_to_handle[begin_idx])
            end_atom = end_parent.fetch_child(atom_idx_to_handle[end_idx])
            begin_residue = _enclosing_role(begin_atom, PrimitiveRole.RESIDUE)
            end_residue = _enclosing_role(end_atom, PrimitiveRole.RESIDUE)
            owner = begin_residue if begin_residue is end_residue else segment

            # Chemistry ownership is role-based: arbitrary grouping nodes preserve
            # organization, while bonds live at RESIDUE or SEGMENT SAAMR owners.
            _connect_rdkit_atom_pair_at_owner(
                rdmol_segment,
                begin_idx,
                end_idx,
                begin_parent,
                atom_idx_to_handle[begin_idx],
                end_parent,
                atom_idx_to_handle[end_idx],
                owner,
                conformer_idx,
                external_linker_label,
                **kwargs,
            )

    if reconstruct_shapes:
        for node in reversed(list(nodes_by_path.values())):
            _rebuild_shape_from_leaf_positions(node)

    return segment


def primitive_from_mupt_sdf(
    source: str | Path | Sequence[str | Path],
    conformer_idx: Optional[int]=0,
    label: Optional[Hashable]=None,
    external_linker_label: str='*',
    reconstruct_bonds: bool=True,
    reconstruct_shapes: bool=True,
    **kwargs,
) -> Primitive:
    """Load a UNIVERSE-rooted Primitive from MuPT-generated SAAMR SDF files.

    Parameters
    ----------
    source : str, pathlib.Path, or sequence of path-like
        One SDF path, a multi-record SDF path, or multiple SDF paths generated by
        :func:`primitive_to_rdkit_mols`.
    conformer_idx : int, optional
        Conformer ID used to transfer atom coordinates. Defaults to ``0`` for SDF.
    label : Hashable, optional
        Label for the returned UNIVERSE root.
    external_linker_label : str, default='*'
        Connector label used when mirroring linker connectors up the hierarchy.
    reconstruct_bonds : bool, default=True
        Rebuild MuPT connector topology from RDKit bonds. Set ``False`` when only
        hierarchy metadata is needed, such as notebook validation before OpenFF
        simulation.
    reconstruct_shapes : bool, default=True
        Aggregate container shapes from atom coordinates after bond reconstruction.

    Returns
    -------
    Primitive
        ``PrimitiveRole.UNIVERSE`` root containing one SEGMENT child per SDF record.

    Raises
    ------
    ValueError
        If the input is not MuPT-generated SAAMR SDF serialization metadata.
    """
    rdmols = _load_mupt_sdf_mols(source)
    universe = Primitive(label=label, role=PrimitiveRole.UNIVERSE)
    universe.metadata.update(_validate_root_metadata_consistency(rdmols))

    # This is a focused issue #48 draft serializer path, not a generic RDKit
    # import fallback. Missing markers mean callers should use primitive_from_rdkit().
    for rdmol in rdmols:
        universe.attach_child(
            _primitive_from_mupt_saamr_mol(
                rdmol,
                conformer_idx=conformer_idx,
                external_linker_label=external_linker_label,
                reconstruct_bonds=reconstruct_bonds,
                reconstruct_shapes=reconstruct_shapes,
                **kwargs,
            )
        )
    return universe


def primitive_from_rdkit_segment(
    rdmol_segment: Mol,
    conformer_idx: Optional[int]=None,
    label: Optional[Hashable]=None,
    residue_role: PrimitiveRole=PrimitiveRole.RESIDUE,
    atom_role: PrimitiveRole=PrimitiveRole.PARTICLE,
    atom_label: str='ATOM',
    external_linker_label: str='*',
    smiles_writer_params: SmilesWriteParams=DEFAULT_SMILES_WRITE_PARAMS,
    **kwargs,
) -> Primitive:
    """Initialize a SAAMR SEGMENT hierarchy from one RDKit Mol.

    Parameters
    ----------
    rdmol_segment : rdkit.Chem.rdchem.Mol
        RDKit molecule representing one connected segment.
    conformer_idx : int, optional
        Conformer ID used to transfer atom coordinates.
    label : Hashable, optional
        Segment label. If absent, MuPT metadata or RDKit naming is used.
    residue_role : PrimitiveRole, default=PrimitiveRole.RESIDUE
        Role assigned to reconstructed residue containers.
    atom_role : PrimitiveRole, default=PrimitiveRole.PARTICLE
        Role assigned to reconstructed atomic Primitives.
    atom_label : str, default='ATOM'
        Child-handle label used when attaching atoms to residues.
    external_linker_label : str, default='*'
        Connector label used when mirroring connectors up the hierarchy.
    smiles_writer_params : SmilesWriteParams, optional
        RDKit SMILES writer settings used for fallback segment naming.

    Returns
    -------
    Primitive
        SEGMENT-role Primitive containing RESIDUE and PARTICLE descendants.
    """
    first_atom = rdmol_segment.GetAtomWithIdx(0) if rdmol_segment.GetNumAtoms() else None
    if label is None:
        if first_atom is not None and first_atom.HasProp("mupt_segment_label"):
            label = first_atom.GetProp("mupt_segment_label")
        else:
            label = name_for_rdkit_mol(rdmol_segment, smiles_writer_params=smiles_writer_params)

    segment_primitive = Primitive(
        label=label,
        metadata=_non_root_metadata(rdmol_segment),
        role=PrimitiveRole.SEGMENT,
    )
    residue_handles: dict[tuple[str, str, int, str], PrimitiveHandle] = {}
    atom_idx_to_residue_handle: dict[int, PrimitiveHandle] = {}
    atom_idx_to_atom_handle: dict[int, PrimitiveHandle] = {}
    linker_idxs: set[int] = set()

    # First pass: rebuild residue containers and attach atomic PARTICLE children.
    for atom in rdmol_segment.GetAtoms():
        atom_idx = atom.GetIdx()
        if is_linker(atom):
            linker_idxs.add(atom_idx)
            continue

        res_key = _residue_key(atom)
        if res_key not in residue_handles:
            _, insertion_code, _, residue_label = res_key
            residue_metadata = {}
            if insertion_code:
                residue_metadata["pdb_insertion_code"] = insertion_code
            residue = Primitive(
                label=residue_label,
                metadata=residue_metadata,
                role=residue_role,
            )
            residue_handles[res_key] = segment_primitive.attach_child(residue)

        residue = segment_primitive.fetch_child(residue_handles[res_key])
        atom_prim = primitive_from_rdkit_atom(
            rdmol_segment,
            atom_idx,
            conformer_idx=conformer_idx,
            attach_connectors=False,
            role=atom_role,
        )
        if atom.HasProp("mupt_particle_label"):
            atom_prim.label = atom.GetProp("mupt_particle_label")
        atom_idx_to_residue_handle[atom_idx] = residue_handles[res_key]
        atom_idx_to_atom_handle[atom_idx] = residue.attach_child(atom_prim, label=atom_label)

    # Second pass: recreate intra-residue bonds or route inter-residue bonds via segment connectors.
    for bond in rdmol_segment.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in linker_idxs or end_idx in linker_idxs:
            if begin_idx in linker_idxs and end_idx in linker_idxs:
                continue

            atom_idx, linker_idx = (
                (end_idx, begin_idx) if begin_idx in linker_idxs else (begin_idx, end_idx)
            )
            residue_handle = atom_idx_to_residue_handle[atom_idx]
            residue = segment_primitive.fetch_child(residue_handle)
            # Linker atoms are not exported as PARTICLEs; their bonds become external
            # connectors so repeat units keep polymer attachment semantics.
            _mirror_linker_bond_to_owner(
                rdmol_segment,
                atom_idx,
                linker_idx,
                residue,
                atom_idx_to_atom_handle[atom_idx],
                segment_primitive,
                conformer_idx,
                external_linker_label,
                **kwargs,
            )
            continue

        begin_res_handle = atom_idx_to_residue_handle[begin_idx]
        end_res_handle = atom_idx_to_residue_handle[end_idx]
        begin_residue = segment_primitive.fetch_child(begin_res_handle)
        end_residue = segment_primitive.fetch_child(end_res_handle)
        owner = begin_residue if begin_res_handle == end_res_handle else segment_primitive
        _connect_rdkit_atom_pair_at_owner(
            rdmol_segment,
            begin_idx,
            end_idx,
            begin_residue,
            atom_idx_to_atom_handle[begin_idx],
            end_residue,
            atom_idx_to_atom_handle[end_idx],
            owner,
            conformer_idx,
            external_linker_label,
            **kwargs,
        )

    # Reconstruct coarse shapes from child atom coordinates when conformer data exists.
    for residue in segment_primitive.children:
        if residue.children:
            _rebuild_shape_from_leaf_positions(residue)

    _rebuild_shape_from_leaf_positions(segment_primitive)
    segment_primitive.check_self_consistent()

    return segment_primitive
    
def primitive_from_rdkit(
    rdmol : Mol,
    conformer_idx : Optional[int]=None,
    label : Optional[Hashable]=None,
    role : PrimitiveRole=PrimitiveRole.UNIVERSE,
    residue_role : PrimitiveRole=PrimitiveRole.RESIDUE,
    atom_role : PrimitiveRole=PrimitiveRole.PARTICLE,
    smiles_writer_params : SmilesWriteParams=DEFAULT_SMILES_WRITE_PARAMS,
    sanitize_frags : bool=True,
    denest : bool=True,
    **kwargs,
) -> Primitive:
    """Initialize a SAAMR hierarchy from an RDKit Mol.

    Parameters
    ----------
    rdmol : rdkit.Chem.rdchem.Mol
        RDKit molecule that may contain one or more disconnected fragments.
    conformer_idx : int, optional
        Conformer ID used to transfer atom coordinates.
    label : Hashable, optional
        Label assigned to the returned root Primitive.
    role : PrimitiveRole, default=PrimitiveRole.UNIVERSE
        Role assigned to the returned UNIVERSE Primitive when ``denest=False``
        or when importing multiple fragments. The legacy single-fragment
        ``denest=True`` path returns ``primitive_from_rdkit_chain(...)`` and uses
        ``residue_role`` for that direct return to preserve the old API shape.
    residue_role : PrimitiveRole, default=PrimitiveRole.RESIDUE
        Role assigned to reconstructed residue containers.
    atom_role : PrimitiveRole, default=PrimitiveRole.PARTICLE
        Role assigned to reconstructed atomic Primitives.
    smiles_writer_params : SmilesWriteParams, optional
        RDKit SMILES writer settings used for fallback segment naming.
    sanitize_frags : bool, default=True
        Passed to RDKit fragment extraction.
    denest : bool, default=True
        Preserve legacy single-fragment direct import when ``True``. Set ``False``
        to always return a UNIVERSE-rooted SAAMR hierarchy.

    Returns
    -------
    Primitive
        Legacy single-fragment Primitive when ``denest=True``; otherwise a
        UNIVERSE-role Primitive with one SEGMENT child per RDKit fragment.
    """
    chains = GetMolFrags(
        rdmol,
        asMols=True, 
        sanitizeFrags=sanitize_frags,
        # DEV: leaving these None for now, but highlighting that we can spigot more info out of this eventually
        frags=None,
        fragsMolAtomMapping=None,
    )

    if (len(chains) == 1) and denest:
        # Compatibility note for review: historically, primitive_from_rdkit(...,
        # denest=True) returned primitive_from_rdkit_chain(...) directly for a
        # single RDKit fragment, while denest=False wrapped fragments under a
        # synthetic root. The role-aware importer would be cleaner if denest were
        # deprecated and every RDKit import returned the canonical SAAMR shape
        # UNIVERSE -> SEGMENT -> RESIDUE -> PARTICLE. That would remove this
        # return-type branch and make downstream role-aware export simpler, but
        # it would break existing callers that rely on the old single-fragment
        # RESIDUE-like Primitive return. Keep the old behavior in this PR so the
        # new RDKit path remains non-breaking; reviewers can decide whether a
        # later PR should formally deprecate denest and migrate callers to the
        # canonical UNIVERSE-rooted import behavior. The caller's role argument is
        # therefore intentionally not applied on this branch; residue_role controls
        # the direct legacy return instead.
        return primitive_from_rdkit_chain(
            chains[0],
            conformer_idx=conformer_idx,
            label=label,
            role=residue_role,
            atom_role=atom_role,
            smiles_writer_params=smiles_writer_params,
            **kwargs,
        )

    universe_primitive = Primitive(
        label=label,
        metadata={
            **_non_root_metadata(rdmol),
            **_root_metadata(rdmol),
        },
        role=role,
    )
    # RDKit fragments are interpreted as separate SEGMENT-role molecules in one UNIVERSE.
    for chain in chains:
        universe_primitive.attach_child(
            primitive_from_rdkit_segment(
                chain,
                conformer_idx=conformer_idx,
                label=None,
                residue_role=residue_role,
                atom_role=atom_role,
                smiles_writer_params=smiles_writer_params,
                **kwargs,
            )
        )
    return universe_primitive
