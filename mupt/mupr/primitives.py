'''Information classes for sets of polymer unit primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Generator, Hashable, Optional, Sequence, Union
from dataclasses import dataclass, field
from enum import Enum # consider having a bitwise Enum to encode possible specification states of a primitive??

import numpy as np

from rdkit import Chem
from rdkit.Chem.rdchem import (
    Mol,
    Conformer,
    ChiralType,
    StereoDescriptor,
    StereoInfo,
    StereoType,
)
from rdkit.Chem.rdmolops import (
    SANITIZE_ALL,
    AROMATICITY_MDL,
)

from .ports import Port
from ..chemistry.sanitization import sanitize_mol
from ..chemistry.selection import (
    bonds_by_condition,
    bond_condition_by_atom_condition_factory,
    atom_is_linker,
    logical_or,
)
from ..geometry.arraytypes import ndarray, N, Shape
from ..geometry.shapes import BoundedShape, PointCloud, Sphere


@dataclass
class Primitive:
    '''Represents a fundamental (but not necessarily irreducible) building block of a polymer system in the abstract 
    
    Note that, by default ALL fields are optional; this is to reflect the fact that use-cases and levels of info provided may vary
    
    For example, one might object that functionality and number of atoms could be derived from the SMILES string and are therefore redundant;
    However, in the case where no chemistry is explicitly provided, it's still perfectly valid to define numbers of atoms present
    E.g. a coarse-grained sticker-and-spacer model
    
    As another example, a 0-functionality primitive is also totally legal (ex. as a complete small molecule in an admixture)
    But comes with the obvious caveat that, in a network, it cannot be incorporated into a larger component
    '''
    num_atoms : Optional[int] = None # number of atoms AS APPEARING ON the periodic, i.e. NOT other generic primitives or virtual linker atoms
    chemistry : Optional[str] = None # line notation specification of chemistry, for now as SMIRKS (numbered SMARTS)
    shape     : Optional[BoundedShape] = None # a rigid shape which approximates and abstracts the behavoir of the primitive in space
    ports     : set[Port] = field(default_factory=set) # a list of ports which are available for bonding to other primitives
    
    stereo_marker : Optional[ChiralType] = None # DEVNOTE: decide if this should be explicit, or be looked for in metadata
    metadata : dict[Hashable, Any] = field(default_factory=dict)
    
    # comparison methods    
    def __hash__(self): # critical that this exists to allow comparison between primitives
        return hash(f'{self.num_atoms}{self.chemistry}{type(self.shape).__name__}{len(self.ports)}')
    
    def __eq__(self, other : object) -> bool:
        '''Check whether two primitives are equivalent'''
        if not isinstance(other, Primitive):
            raise TypeError(f'Cannot compare Primitive to {type(other)}')
        return self.__hash__() == other.__hash__()

    # properties
    @property
    def is_atomic(self) -> bool:
        '''Test if the primitive at hand represents a single atom from the periodic table'''
        ...

    @property
    def has_chemistry(self) -> bool:
        '''Check whether the Primitive has chemical information explicitly specified'''
        return self.chemistry is not None
    
    @property
    def functionality(self) -> int:
        '''Number of neighboring primitives which can be attached to this primitive'''
        return len(self.ports)
    
    # initialization methods    
    def __post_init__(self) -> None:
        self._cleanup()
        
    def _cleanup(self) -> None:
        '''Perform some sanitization based on which inputs are provided and what those inputs are'''
        ...
           
    # TODO: add descriptor for automatically sanitizing and canonicalizing chemistry attr when set
    
    @classmethod
    def from_rdkit(cls, mol : Mol) -> 'Primitive':
        '''Initialize a chemically-explicit Primitive from an RDKit Mol'''
        # clean up RDMol instance
        mol_sanitized = sanitize_mol(
            mol,
            sanitize_ops=SANITIZE_ALL,          # TODO: add option to specify sanitization flags
            aromaticity_model=AROMATICITY_MDL,  # TODO: add option to specify sanitization flags
            in_place=False, # don't modify original Mol instance!
        )
        num_atoms_total = mol_sanitized.GetNumAtoms() # number of atoms INCLUDING virtual linker atoms
        for atom in mol_sanitized.GetAtoms(): # inject map numbers to faithfully preserve atom order
            atom.SetAtomMapNum(atom.GetIdx())
        
        # determine which atoms are "real" and which are "virtual" (linker sites)
        ports : tuple[Port] = tuple(Port.ports_from_rdkit(mol_sanitized)) # convert from generator to tuple (need a Sequence-like type)
        num_ports : int = len(ports)
        
        linker_idxs : list[int] = [port.linker for port in ports] # nneds to be list to properly handle numpy "smart" indexing
        real_atoms_mask = np.ones(num_atoms_total, dtype=bool) # for indexing which atoms are "real" (i.e. not linker atoms)
        real_atoms_mask[linker_idxs] = False # exclude linker atoms from the shape
        
        # determine shape of Primitive; for now either its atomic coordinates (if a conformer is present) or no shape at all
        if (mol_sanitized.GetNumConformers() <= 0): 
            shape = None
        else:
            # for now, will always assume active conformer is the first (idx 0)
            shape = PointCloud(positions=mol_sanitized.GetConformer(0).GetPositions()[real_atoms_mask])
        
        # intialize and return Primitive
        return cls(
            num_atoms=num_atoms_total - num_ports, # exclude "virtual" liker atoms from atom count
            ports=ports,
            shape=shape,
            chemistry=Chem.MolToSmiles(
                mol_sanitized,
                canonical=True,         # canonicalize to allow uniqueness check
                isomericSmiles=True,    # include stereo info
                allHsExplicit=True,     # include all Hs
                allBondsExplicit=True,  # show all bonds
                ignoreAtomMapNumbers=False, # preserve mapped atoms
                doRandom=False,         # make SMILES deterministic
            ),
            stereo_marker=None, # TODO: add stereochemical perception, when atomic
        )
    
    @classmethod
    def from_SMILES(cls, smiles : str, positions : Optional[ndarray[Shape[N, 3], float]]=None) -> 'Primitive':
        '''Initialize a chemically-explicit Primitive from a SMILES string representation of the molecular graph'''
        return cls.from_rdkit(
            mol=Chem.MolFromSmiles(smiles, sanitize=False), # don't mangle molecule with default sanitization - read SMILES verbatim
            positions=positions,    
        )
        
    def to_rdkit(self) -> Mol:
        '''Convert Primitive to an RDKit Mol'''
        if self.chemistry is None:
            raise ValueError('Primitive with no explicit chemistry cannot be exported to RDKit')
        
        mol = sanitize_mol(
            Chem.MolFromSmiles(self.chemistry, sanitize=False), # don't mangle molecule with default sanitization - read SMILES verbatim
            sanitize_ops=SANITIZE_ALL,          # TODO: add option to specify sanitization flags
            aromaticity_model=AROMATICITY_MDL,  # TODO: add option to specify sanitization flags
            in_place=False, # don't modify original Mol instance!
        )
        num_atoms_total = mol.GetNumAtoms() # number of atoms INCLUDING virtual linker atoms
        assert num_atoms_total == self.num_atoms + self.functionality
        
        # TODO: add check on uniqueness and completeness of atom map numbers
        atom_order_map = {
            atom.GetIdx(): atom.GetAtomMapNum()
                for atom in mol.GetAtoms()
        }
        mol = Chem.RenumberAtoms(mol, sorted(atom_order_map, key=atom_order_map.get)) # renumber atoms to be in the order prescribed by the SMIRKS string
        
        # set conformer on RDKit Mol if atom positions are specified
        if isinstance(self.shape, PointCloud):
            positions = np.zeros((num_atoms_total, 3), dtype=float) 
            real_atoms_mask = np.ones(num_atoms_total, dtype=bool) # for indexing which atoms are "real" (i.e. not linker atoms)
            for port in self.ports:
                real_atoms_mask[port.linker] = False 
                if port.linker_position is not None: # only attempt to set linker positions if they are
                    positions[port.linker] = port.linker_position 
            positions[real_atoms_mask] = self.shape.positions
            
            rdconf = Conformer(num_atoms_total)
            rdconf.SetPositions(positions)
            conf_id = mol.AddConformer(rdconf, assignId=0) # always assign to conformer 0 by default
        
        return mol
    
    # interaction methods
    def atomize(self, uniquify: bool=False) -> Generator['Primitive', None, None]:
        '''Decompose primitive into its unique, constituent single-atom primitives'''
        mol = self.to_rdkit()
        real_bond_idxs = bonds_by_condition(
            mol,
            condition=bond_condition_by_atom_condition_factory(
                atom_condition=atom_is_linker, # check if either atom in the bond is a linker
                binary_operator=logical_or,
            ),
            as_indices=True,
            as_pairs=False, 
            negate=True, # exclude all bonds but those between 2 non-linker atoms
        )
        
        seen_primitive_hashes : set[int] = set()
        mol_fragmented : Mol = Chem.FragmentOnBonds(mol, real_bond_idxs, addDummies=True)
        for fragment in Chem.GetMolFrags(mol_fragmented, asMols=True, sanitizeFrags=False):
            for atom in fragment.GetAtoms(): # DEVNOTE: for now, clear all linker flavor markers; eventually, want to keep labels for...
                atom.SetIsotope(0) # ...those which were also linkers in the unfragmented moleucle (won't worry about it for now though)

            sub_primitive = self.from_rdkit(fragment)
            sub_hash = hash(sub_primitive)
            if uniquify and (sub_hash in seen_primitive_hashes):
                continue # skip sub-primitives we've already seen, as requested
            
            seen_primitive_hashes.add(sub_hash)
            yield sub_primitive


@dataclass
class PrimitiveLexicon:
    '''Collection of primitives which form the basis of a'''
    primitives : dict[str, Primitive] = field(default_factory=dict) # priitives keyed by aliases
    
    # this will likely need **some** methods, but most of the sanitization ought to be the responsibility of the MID graph
    # this is more-or-less a glorified container for now
       