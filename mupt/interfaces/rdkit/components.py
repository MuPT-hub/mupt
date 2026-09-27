"""
Utilities for extracting information from and recasting RDKit objects
(e.g. Atom, Bond, Conformer, etc.) and recasting them as MuPT core objects
"""

from typing import (
    Callable,
    Hashable,
    Iterable,
    Optional,
    Type,
)

import numpy as np
from networkx import Graph

from rdkit.Chem.rdchem import Atom, Bond, Mol
from rdkit.Chem.rdmolfiles import MolFragmentToSmarts

# Custom
from ...chemistry.conversion import rdkit_atom_to_element
from ...chemistry.rdkit.selection import (
    AtomCondition,
    logical_or,
    all_atoms,
    atom_neighbors_by_condition,
    bonds_by_condition,
    bond_condition_by_atom_condition_factory,
)

from ...geometry.arraytypes import Vector3, Array2x3
from ...mupr.connection.types import ConnectorLabeller
from ...mupr.connection.connectors import Connector, AttachmentPoint

type AtomLabeller = Callable[[Atom], Hashable]


def DEFAULT_ATOM_LABELLER(atom: Atom) -> str:
    """
    Default implementation of creating a commonly-understood
    hashable label from an RDKit Atom instance
    """
    # return str(atom.GetIdx())
    return f"{rdkit_atom_to_element(atom)!s}-{atom.GetIdx()}"


# Representation component initializers
def chemical_graph_from_rdkit(
    rdmol: Mol,
    atom_condition: Optional[AtomCondition] = None,
    atom_labeller: AtomLabeller = DEFAULT_ATOM_LABELLER,
    binary_operator: Callable[[bool, bool], bool] = logical_or,
    graph_type: Type[Graph] = Graph,
) -> Graph:
    """
    Create a graph from an RDKit Mol whose:
    * Vertices correspond to all atoms satisfying the given atom condition, and
    * Edges are all bonds between the selected atoms

    Parameters
    ----------
    rdmol : Chem.Mol
        The RDKit Mol object to convert.
    atom_condition : Optional[Callable[[Chem.Atom], bool]], default None
        Condition on atoms which returns bool;
        Always returns True if unset
    atom_labeller : Callable[[Chem.Atom], Hashable], default DEFAULT_ATOM_LABELLER
        Method to uniquely label each atom as a vertex in the graph
        Default assignment yields '<element>-<atom index>' strings
    binary_operator : Callable[[bool, bool], bool], default logical_or
        Binary logical operator used to
    """
    _atom_condition = atom_condition if atom_condition else all_atoms
    bond_condition = bond_condition_by_atom_condition_factory(
        _atom_condition, binary_operator
    )

    return graph_type(
        (atom_labeller(atom_begin), atom_labeller(atom_end))
        for (atom_begin, atom_end) in bonds_by_condition(
            rdmol,
            condition=bond_condition,
            as_pairs=True,  # return bond as pair of atoms,
            as_indices=False,  # ...each as Atom objects
            negate=False,
        )
    )


def atom_positions_from_rdkit(
    rdmol: Mol,
    conformer_idx: Optional[int] = None,
    atom_idxs: Optional[Iterable[int]] = None,
) -> Optional[Vector3]:
    """
    Boilerplate for fetching a subset of atom positions
    (if conformer it set) from an RDKit Mol

    Parameters
    ----------
    rdmol : Chem.Mol
        The RDKit Mol object to extract positions from
    conformer_idx : Optional[int], default None
        The ID of the conformer from which to extract 3D positions
        If provided as None, will return None
    atom_idxs : Optional[Iterable[int]], default None
        The indices of a subset of atoms from which to extract positions
        Atom positions will be returned in the same order as the indices provided.

    Returns
    -------
    Optional[np.ndarray[Shape[N, 3], float]]
        The extracted atom positions, returned as an (N, 3) array
        If no conformer index is provided OR if a conformer index is
        provided but no atom indices are provided, returns None instead
    """
    if conformer_idx is None:
        return None

    if atom_idxs is None:
        atom_idxs = (atom.GetIdx() for atom in rdmol.GetAtoms())

    # DEVNOTE: will raise Exception if bad ID is provided; no need to check locally
    # return conformer.GetPositions()[[idx for idx in atom_idxs], :]
    conformer = rdmol.GetConformer(conformer_idx)
    atom_positions = tuple(
        np.array(conformer.GetAtomPosition(atom_idx), dtype=float)
        for atom_idx in atom_idxs
    )
    if atom_positions:
        return np.vstack(atom_positions)
    # making None return explicit just to clarify it can still happen at this stage
    return None


def atom_radius_from_rdkit(atom: Atom) -> Optional[float]:
    """Recover a float-valued atomic radius from an RDKit atom's properties"""
    # TB: append to this as needed
    ATOMIC_RADIUS_KEYS: tuple[str, ...] = ("radius", "rad", "r_LJ")
    for radius_key in ATOMIC_RADIUS_KEYS:
        try:
            return atom.GetDoubleProp(radius_key)
        except KeyError:
            continue
    else:
        return None  # TB: strictly redundant, but added for readability


def connector_from_rdkit_atoms(
    parent_mol: Mol,
    from_atom_idx: int,
    to_atom_idx: int,
    conformer_idx: Optional[int] = None,
    attachables_factory: Callable[[Atom], set[Hashable]] = lambda atom: {atom.GetIdx()},
    connector_labeller: ConnectorLabeller = lambda conn: conn.DEFAULT_LABEL,
    locked: bool = False,
) -> Connector:
    """
    Creates a Connectors representing half of an RDKit Bond,
    spanning from "from_atom" to "to_atom" and inheriting its
    positional, type labelling, bondtype, and other metadata

    Parameters
    ----------
    parent_mol : Mol
        The RDKit Mol containing the atoms of interest
    from_atom_idx : int
        The index of the "anchor" atom of the pair
        on which the Connector will be anchored
    to_atom_idx : int
        The index of the "linker" atom of the pair,
        the neighbor atom of the anchor
    conformer_idx : Optional[int], optional, default None
        The ID of the conformer from which to extract 3D positions
        If None is supplied, will leave all spatial fields of the Connector unset
    attachables_factory : attachables_factory : Callable[[Atom], set[Hashable]], \
            default lambda atom : {atom.GetIdx()},
        A function which takes an RDKit Atom and returns a set of type labels
        to use when initializing an AttachmentPoint for the end of a Connector
    connector_labeller : Callable[[Connector], ConnectorLabel], \
            default: Connector.DEFAULT_LABEL
        A function which takes a Connector object and returns an appropriate label
        This is called after all other fields of the Connector have been set
        (i.e. can make use of those fields in determination of the label)
    locked : bool, default False
        Whether to make the produced Connectors read-only, locking it

    Returns
    -------
    connector : Connector
        The resulting Connector instance
    """
    ## Fetch RDKit Objects
    bond = parent_mol.GetBondBetweenAtoms(from_atom_idx, to_atom_idx)

    anchor_atom = parent_mol.GetAtomWithIdx(from_atom_idx)
    anchor = AttachmentPoint(attachables=attachables_factory(anchor_atom))

    linker_atom = parent_mol.GetAtomWithIdx(to_atom_idx)
    linker = AttachmentPoint(attachables=attachables_factory(linker_atom))

    ## Geometry
    coplanar_point: Optional[Vector3] = None  # defines bond tangent plane, if present
    connector_positions: Optional[Array2x3] = atom_positions_from_rdkit(
        parent_mol,
        conformer_idx=conformer_idx,
        atom_idxs=[from_atom_idx, to_atom_idx],
    )
    if connector_positions is not None:
        anchor.position = connector_positions[0, :]
        linker.position = connector_positions[1, :]

        # define dihedral plane by neighbor atom, if a suitable one is present
        non_neighbor_atom_idxs: Iterable[int] = atom_neighbors_by_condition(
            anchor_atom,
            condition=lambda neighbor: neighbor.GetIdx() == to_atom_idx,
            negate=True,  # ensure the tangent point is not the linker itself
            as_indices=True,
        )
        non_linker_nb_atom_positions = atom_positions_from_rdkit(
            parent_mol,
            conformer_idx=conformer_idx,
            atom_idxs=non_neighbor_atom_idxs,
        )
        if non_linker_nb_atom_positions is not None:
            coplanar_point = non_linker_nb_atom_positions[0, :]

    ## Other data
    metadata = bond.GetPropsAsDict(
        includePrivate=True,
        # NOTE: computed props suppressed to avoid
        # "unpicklable RDKit vector" errors
        includeComputed=False,
    )
    metadata["bond_stereo"] = bond.GetStereo()
    metadata["bond_stereo_atoms"] = tuple(bond.GetStereoAtoms())

    query_smarts = MolFragmentToSmarts(
        parent_mol,
        atomsToUse=[from_atom_idx, to_atom_idx],
        bondsToUse=[bond.GetIdx()],
    )

    ## Final assembly
    connector = Connector(
        anchor=anchor,
        linker=linker,
        bondtype=bond.GetBondType(),
        query_smarts=query_smarts,
        metadata=metadata,
    )
    connector.label = connector_labeller(connector)
    if coplanar_point:
        connector.set_tangent_from_coplanar_point(coplanar_point)

    if locked:
        connector.lock()

    return connector


def connector_pair_from_rdkit_bond(
    parent_mol: Mol,
    bond_idx: int,
    conformer_idx: Optional[int] = None,
    attachables_factory: Callable[[Atom], set[Hashable]] = lambda atom: {atom.GetIdx()},
    connector_labeller: ConnectorLabeller = lambda conn: conn.DEFAULT_LABEL,
    locked: bool = False,
) -> dict[int, Connector]:
    """
    Creates a pair of Connectors representing the two havles of an RDKit Bond
    Created Connectors are pre-assigned as neighbors, and inherit the bonds data:
    * Anchors/linkers are set based on the two bond end atoms
    * Both Connectors' BondType matches that of the bond
    * Positions of each AttachmentPoint will set if conformer information is provided

    Returns a dict, mapping from the atom index of either of the bonds end atoms
    to the created Connector whose anchor is associated with that atom

    Parameters
    ----------
    parent_mol : Mol
        The RDKit Mol containing the atoms of interest
    bond_idx : int
        parent_mol's index for the Bond to be broken apart
    conformer_idx : Optional[int], optional, default None
        The ID of the conformer from which to extract 3D positions
        If None is supplied, will leave all spatial fields of the Connector unset
    *args
        See connector_from_rdkit() docs for remaining args

    Returns
    -------
    idxs_to_connectors : dict[int, Connector]
        A map from anchor atom indices to the correspondingly-anchored Connector
    """
    bond: Bond = parent_mol.GetBondWithIdx(bond_idx)
    begin_atom_idx: int = bond.GetBeginAtomIdx()
    end_atom_idx: int = bond.GetEndAtomIdx()

    ## Assembly
    idxs_to_connectors: dict[int, Connector] = dict()
    for from_atom_idx, to_atom_idx in (
        (begin_atom_idx, end_atom_idx),
        (end_atom_idx, begin_atom_idx),
    ):
        idxs_to_connectors[from_atom_idx] = connector_from_rdkit_atoms(
            parent_mol,
            from_atom_idx=from_atom_idx,
            to_atom_idx=to_atom_idx,
            conformer_idx=conformer_idx,
            attachables_factory=attachables_factory,
            connector_labeller=connector_labeller,
            locked=False,  # don't lock until AFTER connectors are bound as neighbors
        )

    ## Cleanup
    begin_connector = idxs_to_connectors[begin_atom_idx]
    end_connector = idxs_to_connectors[end_atom_idx]

    begin_connector.neighbor = end_connector
    if locked:
        begin_connector.lock()  # mutually locks end_connector also

    return idxs_to_connectors
