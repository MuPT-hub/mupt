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
from rdkit.Chem.rdmolops import (
    FragmentOnBonds,
    GetMolFrags,
    FindPotentialStereo,
)
from rdkit.Chem.rdmolfiles import MolToSmiles, MolFragmentToSmarts

from ..mupr.ports import Port
from ..mupr.primitives import StructuralPrimitive, AtomicPrimitive
from ..mupr.topology import PolymerTopologyGraph

from ..geometry.shapes import PointCloud

from ..chemistry.linkers import LINKER_QUERY_MOL, not_linker, real_and_linker_atom_idxs
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
    conformer : Optional[Conformer] = None
    if (conformer_id is not None): # note: a default conformer_id of -1 actually returns the LAST conformer, not None as we would want
        conformer = rdmol.GetConformer(conformer_id) # will raise Exception if bad ID is provided; no need to check ourselves

    rdmol.UpdatePropertyCache() # avoids implicitValence errors on substructure match
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
            port.linker_position     = conformer.GetAtomPosition(linker_idx)
            port.bridgehead_position = conformer.GetAtomPosition(bh_idx)

            # TODO: offer option to make this more selective (i.e. choose which neighbor atom lies in the dihedral plane)
            for neighbor in bh_atom.GetNeighbors(): # TODO: replace with atom_neighbor_by_condition search
                if not_linker(neighbor) and (neighbor.GetIdx() != linker_idx):
                    port.set_tangent_from_coplanar_point(conformer.GetAtomPosition(neighbor.GetIdx()))
                    break # stop iteration after first valid tangent neighbor is found
                    
        yield port
            
def primitive_from_rdkit(rdmol : Mol, conformer_id : int=Optional[None], label : Optional[Hashable]=None) -> StructuralPrimitive:
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
    # TODO : separate RDKit Mol into distinct connected components (for handling topologies with multiple chains)
    
    # Extract information from the RDKit Mol
    conformer : Optional[Conformer] = None
    if (conformer_id is not None): # note: a default conformer_id of -1 actually returns the LAST conformer, not None as we would want
        conformer = rdmol.GetConformer(conformer_id) # will raise Exception if bad ID is provided; no need to check ourselves
    
    stereo_info_map : dict[int, StereoInfo] = {
        stereo_info.centeredOn : stereo_info # TODO: determine most appropriate choice of flags to use in FindPotentialStereo
            for stereo_info in FindPotentialStereo(rdmol, cleanIt=True, flagPossible=True) 
    }
    real_atom_idxs, linker_idxs = real_and_linker_atom_idxs(rdmol)
    print(linker_idxs)
    # TODO: renumber linkers last? (don't want this done in-place for now)
    
    # 1) Populate bottom-level Primitives from real atoms in RDKit Mol
    external_ports : list[Port] = [] # this is for Ports which do not bond to atoms within the mol
    atomic_primitive_map : dict[int, AtomicPrimitive] = {} # map atom indices to their corresponding Primitive objects for embedding
    
    fragmented_mol = FragmentOnBonds(rdmol, [bond.GetIdx() for bond in rdmol.GetBonds()])
    atom_mol_fragments : tuple[Mol, ...] = GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=False)
    for atom, atom_mol in zip(rdmol.GetAtoms(), atom_mol_fragments):
        atom_idx = atom.GetIdx()
        atom_shape : Optional[PointCloud] = PointCloud(np.array(conformer.GetAtomPosition(atom_idx))) if conformer else None

        ## Collate Port information
        atom_ports : list[Port] = []
        for port in ports_from_rdkit(atom_mol, conformer_id=conformer_id): # NOTE: fragment conformers order and positions that of mirror parent molecule
            atom_ports.append(port)
            if port.linker_flavor in linker_idxs: # TODO: correct linker and bridgehead indices in fragments
                external_ports.append(port) # bonds to linkers constitute Ports which persist at the fragment level
        
        ## assemble atomic-resolution Primitive
        atomic_primitive_map[atom_idx] = AtomicPrimitive(
            structure=atom,
            ports=atom_ports,
            shape=atom_shape,
            label=atom_idx,
            stereo_info=stereo_info_map.get(atom_idx, None),
            metadata=atom.GetPropsAsDict(includePrivate=True), # TODO: set tighter scope for what's included here
        )

    # 2) assemble Primitive at top level  (i.e. at the resolution of the chemical fragment) Primitive
    ## extract topological structure and insert discovered bottom-level atomic Primitives
    # TODO: build this up directly from bonds, rather than remapping the index graph - if not, at least reimplement as embedding
    topology_graph = chemical_graph_from_rdkit( 
        rdmol,
        atom_condition=not_linker,   # only include bonds between two "real" atoms in the topology graph
        binary_operator=logical_and,
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
        
    molecule_shape : Optional[PointCloud] = None
    if conformer is not None:
        molecule_shape = PointCloud(np.array(conformer.GetPositions()[real_atom_idxs]))
    
    return StructuralPrimitive(
        structure=topology_graph,
        ports=external_ports,
        # TODO: find more robust way to make sure this PointCloud stays synchronized w/ the atom postiions
        shape=molecule_shape,
        label=label,
        stereo_info=None,
        metadata=None,
    )
    