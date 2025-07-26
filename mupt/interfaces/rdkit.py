'''Interfaces between MuPT and RDKit Mols'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'


from typing import Hashable, Optional

import networkx as nx
import numpy as np

from rdkit.Chem.rdchem import (
    Atom,
    Bond,
    Mol,
    Conformer,
    StereoInfo
)
from rdkit.Chem.rdmolops import FindPotentialStereo
from rdkit.Chem.rdmolfiles import MolToSmiles

from ..mupr.ports import Port
from ..mupr.primitives import Primitive
from ..mupr.topology import PolymerTopologyGraph

from ..geometry.shapes import PointCloud

from ..chemistry.linkers import is_linker, not_linker
from ..chemistry.selection import atoms_by_condition, logical_and
from ..chemistry.molgraphs import chemical_graph_from_rdkit


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
