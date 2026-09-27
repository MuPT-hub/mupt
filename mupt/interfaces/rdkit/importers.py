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
    AttachablesFactory,
    DEFAULT_ATOM_LABELLER,
    DEFAULT_CONNECTOR_LABELLER,
    DEFAULT_ATTACHABLES_FACTORY,
    atom_positions_from_rdkit,
    atom_radius_from_rdkit,
    connector_from_rdkit_atoms,
    connector_pair_from_rdkit_bond,
)

from ...chemistry.rdkit.labelling import name_for_rdkit_mol
from ...chemistry.rdkit.linkers import is_linker
from ...chemistry.smiles import DEFAULT_SMILES_WRITE_PARAMS, SmilesWriteParams
from ...chemistry.conversion import rdkit_atom_to_element

from ...geometry.shapes import BoundedTransformableShape, PointCloud, Sphere
from ...mupr.primitives import (
    SupportsChildren,
    RootPrimitive,
    CompositePrimitive,
    AtomicPrimitive,
)
from ...mupr.connection.connectors import Connector
from ...builders.heading import TraversalDirection


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
    attachables_factory: AttachablesFactory = DEFAULT_ATTACHABLES_FACTORY,
    connector_labeller: ConnectorLabeller = DEFAULT_CONNECTOR_LABELLER,
    smiles_writer_params: SmilesWriteParams = DEFAULT_SMILES_WRITE_PARAMS,
    **kwargs,
) -> CompositePrimitive:
    """
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
            attachables_factory=attachables_factory,
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
                connector.anchor.attachables.add(chain_direction)
                connector.linker.attachables.add(
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

    ## Assembly
    rdmol_primitive = CompositePrimitive(
        children=atomic_primitives.values(),
        shape=shape,
        metadata=metadata,
        label=label,
    )

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
    else:
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
