"""
Readers which convert RDKit Atoms and Mols
into the MuPT molecular representation
"""

from typing import Any, Hashable, Optional

from rdkit.Chem.rdchem import Atom, Mol
from rdkit.Chem.rdmolops import GetMolFrags

from .components import (
    AtomLabeller,
    ConnectorLabeller,
    MarkersFactory,
    DEFAULT_ATOM_LABELLER,
    DEFAULT_CONNECTOR_LABELLER,
    DEFAULT_ATTACHABLES_FACTORY,
    atom_positions_from_rdkit,
    atom_radius_from_rdkit,
    connector_from_rdkit_atoms,
    connector_pair_from_rdkit_bond,
)

from ...builders.heading import TraversalDirection
from ...geometry.shapes import PointCloud, BoundedTransformableShape, Sphere

from ...chemistry.smiles import DEFAULT_SMILES_WRITE_PARAMS, SmilesWriteParams
from ...chemistry.rdkit.labelling import name_for_rdkit_mol
from ...chemistry.rdkit.linkers import is_linker
from ...chemistry.conversion import rdkit_atom_to_element

from ...mupr.primitives.primitive import SupportsChildren
from ...mupr.primitives.root import RootPrimitive
from ...mupr.primitives.composite import CompositePrimitive
from ...mupr.primitives.simple.atomic import AtomicPrimitive
from ...mupr.connection.connectors import Connector


def primitive_from_rdkit_atom(
    parent_mol: Mol,
    atom_idx: int,
    atom_labeller: AtomLabeller = DEFAULT_ATOM_LABELLER,
    conformer_idx: Optional[int] = None,
    attach_connectors: bool = False,
    **kwargs,
) -> AtomicPrimitive:
    """Initialize an atomic Primitive from an RDKit Atom"""
    atom: Atom = parent_mol.GetAtomWithIdx(atom_idx)
    element = rdkit_atom_to_element(atom)

    ## Shape
    shape: Optional[BoundedTransformableShape] = None

    atom_pos = atom_positions_from_rdkit(
        parent_mol,
        conformer_idx=conformer_idx,
        atom_idxs=[atom_idx],
    )
    if atom_pos is not None:
        center = atom_pos[0, :]
        if (radius := atom_radius_from_rdkit(atom)) is None:
            shape = PointCloud(positions=center)
        else:
            shape = Sphere(radius=radius, center=center)

    ## Other data
    metadata: dict[Hashable, Any] = atom.GetPropsAsDict(
        includePrivate=True,
        # NOTE: computed props suppressed to avoid
        # "unpicklable RDKit vector" errors
        includeComputed=False,
    )
    if (map_num := atom.GetAtomMapNum()) != 0:
        metadata["molAtomMapNumber"] = map_num

    ## Assembly
    atom_primitive = AtomicPrimitive(
        element=element,
        shape=shape,
        metadata=metadata,
        label=atom_labeller(atom),
    )

    ## Connectors (if requested)
    if attach_connectors:
        for nb_atom in atom.GetNeighbors():
            connector = connector_from_rdkit_atoms(
                parent_mol=parent_mol,
                from_atom_idx=atom_idx,
                to_atom_idx=nb_atom.GetIdx(),
                conformer_idx=conformer_idx,
                **kwargs,
            )
            atom_primitive.add_connector(connector)

    return atom_primitive


def primitive_from_rdkit_component(
    rdmol_comp: Mol,
    conformer_idx: Optional[int] = None,
    label: Optional[Hashable] = None,
    atom_labeller: AtomLabeller = DEFAULT_ATOM_LABELLER,
    markers_factory: MarkersFactory = DEFAULT_ATTACHABLES_FACTORY,
    connector_labeller: ConnectorLabeller = DEFAULT_CONNECTOR_LABELLER,
    smiles_writer_params: SmilesWriteParams = DEFAULT_SMILES_WRITE_PARAMS,
) -> CompositePrimitive:
    """
    Initialize a Primitive hierarchy from an RDKit Mol representing a single molecule
        @@ -93,138 +118,105 @@ def primitive_from_rdkit_component(
    label : Hashable, optional
        A distinguishing label for the Primitive
        If none is provided, the canonicalized SMILES of the RDKit Mol will be used
    atom_labeller : Callable[[Chem.Atom], Hashable], default DEFAULT_ATOM_LABELLER
        Method to uniquely label each atom as a vertex in the graph
        Default assignment yields '<element>-<atom index>' strings
    smiles_writer_params: SmilesWriteParams, default DEFAULT_SMILES_WRITE_PARAMS
        Optional configuration how the returned Mol
        is interpreted as a SMILES string by RDKit

    Returns
    -------
    mol_primitive : CompositePrimitive
        The created Primitive object
    """
    ## Compile Atmic sub-Primitives
    non_linker_idxs: list[int] = []  # important that this be ordered, i.e. not a set
    linker_idxs: set[int] = set()  # need to keep track for linker removal pre-hierarchy
    atomic_primitives: dict[int, AtomicPrimitive] = dict()

    for atom in rdmol_comp.GetAtoms():
        atom_idx = atom.GetIdx()
        if is_linker(atom):
            linker_idxs.add(atom_idx)
            continue  # don't map placeholder atoms (slight memory savings)

        non_linker_idxs.append(atom_idx)
        atomic_primitives[atom_idx] = primitive_from_rdkit_atom(
            parent_mol=rdmol_comp,
            atom_idx=atom_idx,
            atom_labeller=atom_labeller,
            conformer_idx=conformer_idx,
            attach_connectors=False,  # will be done in subsequent step
        )

    ## Compile Connections
    for bond in rdmol_comp.GetBonds():
        # pre-links Connector pairs along each bond; no extra work necessary
        idxs_to_connectors: dict[int, Connector] = connector_pair_from_rdkit_bond(
            rdmol_comp,
            bond_idx=bond.GetIdx(),
            conformer_idx=conformer_idx,
            markers_factory=markers_factory,
            locked=False,
        )
        for atom_idx, connector in idxs_to_connectors.items():
            if atom_idx in linker_idxs:
                # leave counterpart one "real" atom unbound, as expected
                del connector.neighbor
                continue

            atom = rdmol_comp.GetAtomWithIdx(atom_idx)
            if (mapnum := atom.GetAtomMapNum()) in {1, 2}:
                # TB: change to looks from isotope eventually, opening up more values?
                chain_direction = TraversalDirection(mapnum)
                connector.anchor.markers.add(chain_direction)
                connector.linker.markers.add(
                    TraversalDirection.complement(chain_direction)
                )

            atomic_primitives[atom_idx].add_connector(
                connector,
                label=connector_labeller,
            )

    ## Geometry
    shape: Optional[BoundedTransformableShape] = None

    non_linker_positions = atom_positions_from_rdkit(
        rdmol_comp,
        conformer_idx=conformer_idx,
        atom_idxs=non_linker_idxs,
    )
    if non_linker_positions is not None:
        shape = PointCloud(positions=non_linker_positions)
        # TODO: add capability to do Ellipsoid/Rod sizing here

    ## Other data
    if label is None:
        label = name_for_rdkit_mol(rdmol_comp, smiles_writer_params)
    metadata = rdmol_comp.GetPropsAsDict(
        includePrivate=True,
        includeComputed=False,
    )
    ## TB: opting to not inject stereochemical metadata for now,
    ## since that may change as Primitive repr is transformed geometrically
    # stereo_info_map : dict[int, StereoInfo] = {
    #     stereo_info.centeredOn : stereo_info
    #        for stereo_info in FindPotentialStereo(
    #            rdmol_comp,
    #            cleanIt=True,
    #            flagPossible=True,
    #        )
    # }

    # 1) Insert child Primitives for each atom (EVEN linkers -
    # this keeps indices in sync for final handle assignment)
    ## DEV: as-implemented, handle idx **SHOULD** match
    ## atom idx, but it never hurts to be explicit :P
    linker_idxs: set[int] = set()
    atom_idx_to_handle_map: dict[int, PrimitiveHandle] = dict()

    # DEV: opting not to get atoms implicitly from bonds to
    # handle single, unbonded atom (e.g. noble gas) uniformly
    for atom in rdmol_chain.GetAtoms():
        atom_idx = atom.GetIdx()
        if is_linker(atom):
            linker_idxs.add(atom_idx)

        atom_prim = primitive_from_rdkit_atom(
            rdmol_chain,
            atom_idx,
            conformer_idx=conformer_idx,
            # will attach per-bond to avoid matching connector handles to bond idxs
            attach_connectors=False,
        )
        atom_idx_to_handle_map[atom_idx] = rdmol_primitive.attach_child(
            atom_prim, label=atom_label
        )

    # 2) forge connections between Primitives corresponding to bonded atoms
    # (propagating external Connectors up to mol primitive)
    for bond in rdmol_chain.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        # Primitive 1 + associated Connector
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
        rdmol_primitive.bind_external_connector(
            begin_prim_handle, begin_conn_handle, label=external_linker_label
        )

        # Primitive 2 + associated Connector
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
        rdmol_primitive.bind_external_connector(
            end_prim_handle, end_conn_handle, label=external_linker_label
        )

        # joining of the pair of Connectors
        rdmol_primitive.connect_children(
            begin_prim_handle,
            begin_conn_handle,
            end_prim_handle,
            end_conn_handle,
        )

    # 3) excise temporary linker Primitives no longer needed as doorstops
    for linker_idx in linker_idxs:
        rdmol_primitive.detach_child(atom_idx_to_handle_map[linker_idx])

    # 3a) insert traversal direction info based on 1-2 map number convention
    for ext_conn_handle, conn_ref in rdmol_primitive.external_connectors.items():
        atom_primitive = rdmol_primitive.fetch_child(conn_ref.primitive_handle)
        ext_conn = rdmol_primitive.fetch_connector(ext_conn_handle)

        if (mapnum := atom_primitive.metadata.get("molAtomMapNumber")) in {1, 2}:
            chain_direction = TraversalDirection(mapnum)
            ext_conn.anchor.markers.add(chain_direction)
            ext_conn.linker.markers.add(TraversalDirection.complement(chain_direction))

    # 4) Inject conformer info
    ## DEV: there are many avenues to do this
    ## (e.g. collate shape from children, if not None on all),
    ## but opted for the simplest for now
    non_linker_conformer = atom_positions_from_rdkit(
        rdmol_chain,
        conformer_idx=conformer_idx,
        atom_idxs=sorted(
            atom_idx_to_handle_map.keys() - linker_idxs
        ),  # preserve atom order
    )
    if (
        non_linker_conformer is not None
    ):  # can't just check if Falsy in case this is an array (would need all() then)
        rdmol_primitive.shape = PointCloud(
            positions=non_linker_conformer
        )  # exploit default NoneType value
    rdmol_primitive.check_self_consistent()

    return rdmol_primitive


def primitive_from_rdkit(
    rdmol: Mol,
    conformer_idx: Optional[int] = None,
    label: Optional[Hashable] = None,
    smiles_writer_params: SmilesWriteParams = DEFAULT_SMILES_WRITE_PARAMS,
    sanitize_frags: bool = True,
    denest: bool = True,
    **kwargs,
) -> SupportsChildren:
    """
    Initialize a Primitive hierarchy from an
    RDKit Mol representing one or more molecules
    """
    chains = GetMolFrags(
        rdmol,
        asMols=True,
        sanitizeFrags=sanitize_frags,
        # DEV: leaving these None for now, but highlighting
        # that we can spigot more info out of this eventually
        frags=None,
        fragsMolAtomMapping=None,
    )

    if (len(chains) == 1) and denest:
        return primitive_from_rdkit_component(
            chains[0],
            conformer_idx=conformer_idx,
            label=label,
            smiles_writer_params=smiles_writer_params,
            **kwargs,
        )

    # DEV: deliberately excluding metadata here to not crowd out per-mol metadata
    universe_primitive = RootPrimitive(label=label)
    for chain in chains:
        universe_primitive.attach_child(
            primitive_from_rdkit_component(
                chain,
                conformer_idx=conformer_idx,
                label=None,  # impose default label for each individual chain
                smiles_writer_params=smiles_writer_params,
                **kwargs,
            )
        )
    return universe_primitive
