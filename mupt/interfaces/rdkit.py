'''Interfaces between MuPT and RDKit Mols'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'


from typing import Callable, Generator, Hashable, Optional, Type, TypeVar

import numpy as np
import networkx as nx
GraphLike = TypeVar('GraphLike', bound=nx.Graph)

from rdkit.Chem.rdchem import (
    Atom,
    Bond,
    Mol,
    Conformer,
    StereoInfo
)
from rdkit.Chem.rdmolops import FindPotentialStereo
from rdkit.Chem.rdmolfiles import MolToSmiles, MolFragmentToSmarts

from ..mupr.ports import Port
from ..mupr.primitives import Primitive
from ..mupr.topology import PolymerTopologyGraph

from ..geometry.shapes import PointCloud

from ..chemistry.linkers import is_linker, not_linker, LINKER_QUERY_MOL
from ..chemistry.selection import (
    AtomCondition,
    BondCondition,
    logical_or,
    logical_and,
    all_atoms,
    atoms_by_condition,
    bonds_by_condition,
    bond_condition_by_atom_condition_factory
)


def chemical_graph_from_rdkit(
    rdmol : Mol,
    atom_condition : Optional[AtomCondition]=None,
    label_method : Callable[[Atom], Hashable]=lambda atom : atom.GetIdx(),
    binary_operator : Callable[[bool, bool], bool]=logical_or,
    graph_type : Type[GraphLike]=nx.Graph,
) -> GraphLike:
    '''
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
    label_method : Callable[[Chem.Atom], Hashable], default lambda atom : atom.GetIdx()
        Method to uniquely label each atom as a vertex in the graph
        Default to choosing the atom's index
    binary_operator : Callable[[bool, bool], bool], default logical_or
        Binary logical operator used to 
    '''
    if not atom_condition:
        atom_condition : AtomCondition = all_atoms
    bond_condition : BondCondition = bond_condition_by_atom_condition_factory(atom_condition, binary_operator)

    return graph_type(
        (label_method(atom_begin), label_method(atom_end))
            for (atom_begin, atom_end) in bonds_by_condition(
                rdmol,
                condition=bond_condition,
                as_pairs=True,      # return bond as pair of atoms,
                as_indices=False,   # ...each as Atom objects
                negate=False,
            )
    )

def ports_from_rdkit(rdmol : Mol, conformer_id : Optional[int]=None) -> Generator['Port', None, None]:
    '''Determine all Ports contained in an RDKit Mol, as specified by wild-type linker atoms'''
    # Extract information from the RDKit Mol
    conformer : Optional[Conformer] = None
    if (conformer_id is not None): # note: a default conformer_id of -1 actually returns the LAST conformer, not None as we would want
        conformer = rdmol.GetConformer(conformer_id) # will raise Exception if bad ID is provided; no need to check ourselves
        positions = conformer.GetPositions()

    for (linker_idx, bh_idx) in rdmol.GetSubstructMatches(LINKER_QUERY_MOL, uniquify=False): # DON'T de-duplify indices (fails to catch both ports on a neutronium)
        linker_atom : Atom = rdmol.GetAtomWithIdx(linker_idx)
        bh_atom     : Atom = rdmol.GetAtomWithIdx(bh_idx)
        port_bond   : Bond = rdmol.GetBondBetweenAtoms(bh_idx, linker_idx)

        port = Port(
            linker=linker_idx, # for now, assign the index to allow easy reverse-lookup of the atom
            bridgehead=bh_idx,
            bondtype=port_bond.GetBondType(),
            linker_flavor=linker_atom.GetIsotope(),
            query_smarts=MolFragmentToSmarts(
                rdmol,
                atomsToUse=[linker_idx, bh_idx],
                bondsToUse=[port_bond.GetIdx()],
            )
        )
        
        if conformer: # solicit coordinates, if available
            port.linker_position     = positions[linker_idx]
            port.bridgehead_position = positions[bh_idx]

            # TODO: offer option to make this more selective (i.e. choose which neighbor atom lies in the dihedral plane)
            for neighbor in bh_atom.GetNeighbors(): # TODO: replace with atom_neighbor_by_condition search
                if neighbor.GetAtomicNum() > 0: # take first real neighbor atom for now
                    port.set_tangent_from_coplanar_point(positions[neighbor.GetIdx()])
                    break
                    
        yield port
            
def primitive_from_rdkit(rdmol : Mol, conformer_id : int=Optional[None], label : Optional[Hashable]=None) -> Primitive:
    """ 
    Create a Primitive with chemically-accuracy ports and internal structure from an RDKit Mol
    
    Parameters
    ----------
    rdmol : Chem.Mol
        The RDKit Mol object to convert.
    conformer_id : int, optional
        The ID of the conformer to use, by default None (uses no conformer)
    label : Hashable, optional
        A distinguishing label for the Primitive
        If none is provided, the canonical SMILES of the RDKit Mol will be used
    
    Returns
    -------
    Primitive
        The created Primitive object.
    """
    # Extract information from the RDKit Mol
    conformer : Optional[Conformer] = None
    if (conformer_id is not None): # note: a default conformer_id of -1 actually returns the LAST conformer, not None as we would want
        conformer = rdmol.GetConformer(conformer_id) # will raise Exception if bad ID is provided; no need to check ourselves

    stereo_info_map : dict[int, StereoInfo] = {
        stereo_info.centeredOn : stereo_info
            for stereo_info in FindPotentialStereo(rdmol, cleanIt=True, flagPossible=True) # TODO: determine most appropriate choice of flags to use here
    }
    # TODO: renumber linkers last? 
    
    # 1) Populate bottom-level Primitives from real atoms in RDKit Mol
    external_ports : list[Port] = [] # this is for Ports which do not bond to atoms within the mol
    atomic_primitive_map : dict[int, Primitive] = {} # map atom indices to their corresponding Primitive objects
    for atom in atoms_by_condition(rdmol, condition=not_linker, as_indices=False, negate=False):
        atom_idx = atom.GetIdx()
        atom_position : Optional[np.ndarray] = np.array(conformer.GetAtomPosition(atom_idx)) if conformer else None
        
        ## Collate Port information
        atom_ports : list[Port] = []
        for neighbor in atom.GetNeighbors():
            nb_idx = neighbor.GetIdx()
            neighbor_port = Port(
                bridgehead=atom_idx,
                linker=nb_idx,
                bondtype=rdmol.GetBondBetweenAtoms(atom_idx, nb_idx).GetBondType(),
                bridgehead_position=atom_position,
                linker_position=np.array(conformer.GetAtomPosition(nb_idx)) if conformer else None,
                # TODO: set tangent position from neighbor - decide upon rules for choosing which neighbor to pick for the dihedral plane
            )
            
            atom_ports.append(neighbor_port)
            if is_linker(neighbor): # bonds to linkers constitute Ports which persist at the fragment level
                external_ports.append(neighbor_port)
        
        ## assemble atomic-resolution Primitive
        atomic_primitive_map[atom_idx] = Primitive(
            structure=atom,
            ports=atom_ports,
            shape=PointCloud(atom_position) if conformer else None,
            label=atom_idx,
            stereo_info=stereo_info_map.get(atom_idx, None),
            metadata=atom.GetPropsAsDict(includePrivate=True), # TODO: set tighter scope for what's included here
        )
    real_atom_idxs : list[int] = list(atomic_primitive_map.keys())

    # 2) assemble Primitive at top level  (i.e. at the resolution of the chemical fragment) Primitive
    ## extract topological structure and insert discovered bottom-level atomic Primitives
    topology_graph = chemical_graph_from_rdkit( # TODO: build this up directly, rather than remapping the index graph
        rdmol,
        atom_condition=not_linker, # only include "real" atoms in the topology graph
        binary_operator=logical_and, # only include node if BOTH atoms are not linkers
        graph_type=PolymerTopologyGraph,
    )
    topology_graph = nx.relabel_nodes(topology_graph, atomic_primitive_map, copy=True)
    
    ## determine label from canonical SMILES, if none is given
    if label is None:
        label = MolToSmiles( # TODO: add some kind of index mixin to distinguish copies of a molecule or chemical fragment
            rdmol, 
            isomericSmiles=True,
            kekuleSmiles=False,
            canonical=True, # this is the critical one!
            allHsExplicit=False,
            doRandom=False,        
        )
    
    return Primitive(
        structure=topology_graph,
        ports=external_ports,
        # TODO: find more robust way to make sure this PointCloud stays synchronized w/ the atom postiions
        shape=PointCloud(np.array(conformer.GetPositions()[real_atom_idxs]) if conformer else None),
        label=label,
        stereo_info=None,
        metadata=None,
    )
    