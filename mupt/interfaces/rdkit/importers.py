'''Readers which convert RDKit Atoms and Mols into the MuPT molecular representation'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'

from typing import (
    Hashable,
    Optional,
)

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
    MUPT_ROOT_METADATA_COUNT,
    MUPT_ROOT_METADATA_KEY_PREFIX,
    MUPT_ROOT_METADATA_VALUE_PREFIX,
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


def _apply_traversal_direction(connector, atom_primitive: Primitive) -> None:
    """Annotate linker connectors from RDKit map-number direction markers."""
    if (mapnum := atom_primitive.metadata.get('molAtomMapNumber')) in {1, 2}:
        chain_direction = TraversalDirection(mapnum)
        connector.anchor.attachables.add(chain_direction)
        connector.linker.attachables.add(TraversalDirection.complement(chain_direction))


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
            atom_handle = atom_idx_to_atom_handle[atom_idx]
            atom_primitive = residue.fetch_child(atom_handle)
            atom_conn_handle = atom_primitive.register_connector(
                connector_between_rdatoms(
                    rdmol_segment,
                    from_atom_idx=atom_idx,
                    to_atom_idx=linker_idx,
                    conformer_idx=conformer_idx,
                    **kwargs,
                )
            )
            # Linker atoms are not exported as PARTICLEs; their bonds become external
            # connectors so repeat units keep polymer attachment semantics.
            residue_conn_handle = residue.bind_external_connector(
                atom_handle,
                atom_conn_handle,
                label=external_linker_label,
            )
            segment_conn_handle = segment_primitive.bind_external_connector(
                residue_handle,
                residue_conn_handle,
                label=external_linker_label,
            )
            for connector in (
                atom_primitive.fetch_connector(atom_conn_handle),
                residue.fetch_connector(residue_conn_handle),
                segment_primitive.fetch_connector(segment_conn_handle),
            ):
                _apply_traversal_direction(connector, atom_primitive)
            continue

        begin_res_handle = atom_idx_to_residue_handle[begin_idx]
        end_res_handle = atom_idx_to_residue_handle[end_idx]
        begin_residue = segment_primitive.fetch_child(begin_res_handle)
        end_residue = segment_primitive.fetch_child(end_res_handle)

        begin_atom_handle = atom_idx_to_atom_handle[begin_idx]
        begin_atom = begin_residue.fetch_child(begin_atom_handle)
        begin_conn_handle = begin_atom.register_connector(
            connector_between_rdatoms(
                rdmol_segment,
                from_atom_idx=begin_idx,
                to_atom_idx=end_idx,
                conformer_idx=conformer_idx,
                **kwargs,
            )
        )
        begin_res_conn_handle = begin_residue.bind_external_connector(
            begin_atom_handle,
            begin_conn_handle,
            label=external_linker_label,
        )

        end_atom_handle = atom_idx_to_atom_handle[end_idx]
        end_atom = end_residue.fetch_child(end_atom_handle)
        end_conn_handle = end_atom.register_connector(
            connector_between_rdatoms(
                rdmol_segment,
                from_atom_idx=end_idx,
                to_atom_idx=begin_idx,
                conformer_idx=conformer_idx,
                **kwargs,
            )
        )
        end_res_conn_handle = end_residue.bind_external_connector(
            end_atom_handle,
            end_conn_handle,
            label=external_linker_label,
        )

        if begin_res_handle == end_res_handle:
            # Bonds within one residue are internal to the RESIDUE Primitive.
            begin_residue.connect_children(
                begin_atom_handle,
                begin_conn_handle,
                end_atom_handle,
                end_conn_handle,
            )
        else:
            # Cross-residue bonds are mirrored up to the SEGMENT before connecting residues.
            segment_primitive.bind_external_connector(
                begin_res_handle,
                begin_res_conn_handle,
                label=external_linker_label,
            )
            segment_primitive.bind_external_connector(
                end_res_handle,
                end_res_conn_handle,
                label=external_linker_label,
            )
            segment_primitive.connect_children(
                begin_res_handle,
                begin_res_conn_handle,
                end_res_handle,
                end_res_conn_handle,
            )

    # Reconstruct coarse shapes from child atom coordinates when conformer data exists.
    for residue in segment_primitive.children:
        if residue.children:
            positions = [atom.shape.centroid for atom in residue.children if atom.shape is not None]
            if positions:
                residue.shape = PointCloud(positions=positions)

    positions = [atom.shape.centroid for atom in segment_primitive.leaves if atom.shape is not None]
    if positions:
        segment_primitive.shape = PointCloud(positions=positions)
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
