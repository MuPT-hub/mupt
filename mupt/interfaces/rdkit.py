'''Interfaces between MuPT and RDKit Mols'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'


from typing import Callable, Generator, Hashable, Optional, Type, TypeVar, Union

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
    AddHs,
    FragmentOnBonds,
    FindPotentialStereo,
    GetMolFrags,
    SanitizeMol,
    SanitizeFlags,
    SANITIZE_ALL,
)
from rdkit.Chem.rdmolfiles import (
    MolFromSmiles,
    MolToSmiles,
    MolFragmentToSmarts,
)
from rdkit.Chem.rdDistGeom import EmbedMolecule

from ..mupr.ports import Port
from ..mupr.primitives import StructuralPrimitive, AtomicPrimitive
from ..mupr.topology import PolymerTopologyGraph
from ..mupr.embedding import embed_primitive_topology

from ..geometry.shapes import PointCloud

from ..chemistry.linkers import (
    not_linker,
    anchor_and_linker_idxs,
    real_and_linker_atom_idxs,
)
from ..chemistry.selection import (
    AtomCondition,
    BondCondition,
    logical_or,
    logical_and,
    all_atoms,
    atoms_by_condition,
    atom_neighbors_by_condition,
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
                as_pairs=True,    # return bond as pair of atoms,
                as_indices=False, # ...each as Atom objects
                negate=False,
            )
    )

def ports_from_rdkit(
        rdmol : Mol,
        conformer_id : Optional[int]=None,
        linker_labeller : Callable[[Atom], int]=lambda atom : atom.GetIdx(),
        anchor_labeller : Callable[[Atom], int]=lambda atom : atom.GetIdx(),
    ) -> Generator['Port', None, None]:
    '''Determine all Ports contained in an RDKit Mol, as specified by wild-type linker atoms'''
    conformer : Optional[Conformer] = None
    if (conformer_id is not None):
        conformer = rdmol.GetConformer(conformer_id)
        positions : np.ndarray = conformer.GetPositions() 

    rdmol.UpdatePropertyCache() # avoids implicitValence errors on substructure match
    for (anchor_idx, linker_idx) in anchor_and_linker_idxs(rdmol):
        linker_atom : Atom = rdmol.GetAtomWithIdx(linker_idx)
        anchor_atom : Atom = rdmol.GetAtomWithIdx(anchor_idx)
        port_bond : Bond = rdmol.GetBondBetweenAtoms(anchor_idx, linker_idx)

        port = Port(
            anchor=anchor_labeller(anchor_atom),
            linker=linker_labeller(linker_atom),
            bondtype=port_bond.GetBondType(),
            query_smarts=MolFragmentToSmarts(
                rdmol,
                atomsToUse=[linker_idx, anchor_idx],
                bondsToUse=[port_bond.GetIdx()],
            )
        )
        
        if conformer:
            port.linker_position = positions[linker_idx, :]
            port.anchor_position = positions[anchor_idx, :]

            # define dihedral plane by neighbor atom, if a suitable one is present
            real_neighbor_atom_idxs : Generator[int, None, None] = atom_neighbors_by_condition(
                anchor_atom,
                condition=lambda neighbor : (neighbor.GetIdx() == linker_idx),
                negate=True, # ensure the tangent point is not the linker itself
                as_indices=True,
            )
            try:
                ## TODO: offer option to make this more selective (i.e. choose which neighbor atom lies in the dihedral plane)
                port.set_tangent_from_coplanar_point(positions[next(real_neighbor_atom_idxs), :])
            except StopIteration:
                pass

        yield port
            
def primitive_from_rdkit(
        rdmol : Mol,
        conformer_id : int=Optional[None],
        label : Optional[Hashable]=None,
    ) -> StructuralPrimitive:
    ''' 
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
    '''
    # TODO : separate RDKit Mol into distinct connected components (for handling topologies with multiple chains)
    
    # Extract information from the RDKit Mol
    conformer : Optional[Conformer] = None
    if (conformer_id is not None): # NOTE: a default conformer_id of -1 actually returns the LAST conformer, not None as we would want
        conformer = rdmol.GetConformer(conformer_id) # DEVNOTE: will raise Exception if bad ID is provided; no need to check ourselves
        positions : np.ndarray = conformer.GetPositions() 
    
    stereo_info_map : dict[int, StereoInfo] = {
        stereo_info.centeredOn : stereo_info # TODO: determine most appropriate choice of flags to use in FindPotentialStereo
            for stereo_info in FindPotentialStereo(rdmol, cleanIt=True, flagPossible=True) 
    }
    real_atom_idxs, external_linker_idxs = real_and_linker_atom_idxs(rdmol)
    # TODO: renumber linkers last? (don't want this done in-place for now)
    
    # Populate bottom-level Primitives from real atoms in RDKit Mol
    external_ports : list[Port] = [] # this is for Ports which do not bond to atoms within the mol
    atomic_primitive_map : dict[int, AtomicPrimitive] = {} # map atom indices to their corresponding Primitive objects for embedding
    
    fragmented_mol = FragmentOnBonds(rdmol, [bond.GetIdx() for bond in rdmol.GetBonds()], addDummies=True) # record linker atom index as dummy isotope
    atom_mol_fragments : tuple[Mol, ...] = GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=False)
    for atom, atom_mol in zip(rdmol.GetAtoms(), atom_mol_fragments):
        atom_idx = atom.GetIdx()

        atom_ports : list[Port] = []
        for port in ports_from_rdkit(
                atom_mol,
                conformer_id=conformer_id, # NOTE: fragment conformers order and positions that of mirror parent molecule
                linker_labeller=lambda a : a.GetIsotope(), # read linker label off of dummy atom
                anchor_labeller=lambda _ : atom_idx,   # by definition, this atom is the anchor of all Ports attached to the atom
            ): 
            atom_ports.append(port)
            if port.linker in external_linker_idxs:
                external_ports.append(port)
        
        atom_shape : Optional[PointCloud] = None
        if conformer:
            atom_shape = PointCloud(positions[atom_idx, :])

        atomic_primitive_map[atom_idx] = AtomicPrimitive(
            structure=atom,
            ports=atom_ports,
            shape=atom_shape,
            label=atom_idx,
            metadata={
                **atom.GetPropsAsDict(includePrivate=True),
                'stereo_info' : stereo_info_map.get(atom_idx, None)
            }, 
        )

    # Assemble Primitive at top level (i.e. at the resolution of the chemical fragment) Primitive
    # DEVNOTE: consider building this up directly from bonds, rather than remapping the index graph
    topology_graph = embed_primitive_topology(
        topology=chemical_graph_from_rdkit( 
            rdmol,
            atom_condition=not_linker,   
            binary_operator=logical_and, # only include bonds where BOTH atoms are "real"
            graph_type=PolymerTopologyGraph,
        ),
        mapping=atomic_primitive_map,
    )
    
    if label is None:
        label = MolToSmiles(
            rdmol, 
            isomericSmiles=True,
            kekuleSmiles=False,
            canonical=True, # this is the critical one!
            allHsExplicit=False,
            doRandom=False,        
        )  # TODO: add some kind of index mixin to distinguish copies of a molecule or chemical fragment
        
    molecule_shape : Optional[PointCloud] = None
    if conformer is not None:
        molecule_shape = PointCloud(positions[real_atom_idxs])

    return StructuralPrimitive(
        structure=topology_graph,
        ports=external_ports,
        shape=molecule_shape,
        label=label,
        metadata=None,
    )
    
def primitive_to_rdkit(primitive : Union[AtomicPrimitive, StructuralPrimitive]) -> Mol:
    '''Convert a StructuralPrimitive to an RDKit Mol'''
    if not primitive.is_all_atom:
        raise ValueError('Cannot export Primitive with non-atomic parts to RDKit Mol')

    # handle identification of shape (if EVERY atom has a postiions and if those are consistent with the Primitive's shape, if PointCloud)

    # case 1 : export single atom to RDKit Atom
    
    # case 2 : link up atoms within strucutral primitive recursively
    ## match ports along bonds, identify external ports
    
# SMILES/SMARTS readers and writers
def primitive_from_smiles(
        smiles : str, 
        ensure_explicit_Hs : bool=True,
        embed_positions : bool=False,
        sanitize_ops : SanitizeFlags=SANITIZE_ALL,
        label : Optional[Hashable]=None,
    ) -> StructuralPrimitive:
    '''Create a Primitive from a SMILES string, optionally embedding positions if selected'''
    rdmol = MolFromSmiles(smiles, sanitize=False)
    if ensure_explicit_Hs:
        rdmol.UpdatePropertyCache() # allow Hs to be added without sanitizating twice
        rdmol = AddHs(rdmol)
    SanitizeMol(rdmol, sanitizeOps=sanitize_ops)
    
    conformer_id : Optional[int] = None
    if embed_positions:
        conformer_id = EmbedMolecule(rdmol, clearConfs=False) # NOTE: don't clobber existing conformers for safety (though new Mol shouldn't have any anyway)
    
    return primitive_from_rdkit(rdmol, conformer_id=conformer_id, label=label)