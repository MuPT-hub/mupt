'''Information classes for sets of polymer unit primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Hashable, Optional, Union
from dataclasses import dataclass, field
from enum import Enum # consider having a bitwise Enum to encode possible specification states of a primitive??

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

from ..geometry.arraytypes import ndarray, N, Shape
from ..geometry.shapes import BoundedShape, PointCloud, Sphere

from ..chemistry.sanitization import sanitize_mol
from ..chemistry.linkers import get_num_linkers


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
    # excluding naming for now as this may screw up comparison of otherwise identical primitives (maybe we'd want that down the line?)
    num_atoms     : Optional[int] = None # number of atoms (these are precisely periodic-table atoms, NOT other generic particles)
    functionality : Optional[int] = None # number of linker sites, which can connect to other primitives
    shape         : Optional[BoundedShape] = None # a rigid shape which approximates and abstracts the behavoir of the primitive in space
    
    # would also be cool to have chemistry-free method of labelling ports (perhaps passing Sequence of Port-type objects in addition to chemistry?)
    chemistry     : Optional[Mol] = None # a SMILES-like string which represents the internal chemistry of the primitive
    smiles        : Optional[str] = None #field(init=False, default=None)
    stereo_marker : Optional[ChiralType] = None
    
    metadata : dict[Hashable, Any] = field(default_factory=dict)
    
    # comparison methods    
    def __hash__(self):
        raise NotImplemented # critical that this exists to allow comparison between primitives
    
    def __eq__(self, value):
        raise NotImplemented # chemical equality will likely be the hardest part to compare - SMILES canonicalization might alleviate this?

    # properties
    @property
    def is_atomic(self) -> bool:
        '''Test if the primitive at hand represents a single atom from the periodic table'''
        ...

    @property
    def has_chemistry(self) -> bool:
        '''Check whether the Primitive has chemical information explicitly specified'''
        return self.chemistry is not None
    
    # initialization methods    
    def __post_init__(self) -> None:
        self._cleanup()
        
    def _cleanup(self) -> None:
        '''Perform some sanitization based on which inputs are provided and what those inputs are'''
        ...
        
    def set_positions(self, positions : Optional[ndarray[Shape[N, 3], float]]) -> None:
        '''Set Primitive shape as an array of coordinates, and (if chemistry is explicit) on the internal Mol representation'''
        if positions is None: #if positions is None:
            return # TODO: handle null positions case monadically?
        
        self.shape = PointCloud(coordinates=positions)
        if self.has_chemistry and isinstance(self.chemistry, Mol):
            n_atoms = self.chemistry.GetNumAtoms()
            assert (n_atoms == len(positions)) # TODO: have RDKit conformer setting account for null atoms/ports

            rdconf = Conformer(n_atoms)
            rdconf.SetPositions(positions)
            conf_id = self.chemistry.AddConformer(rdconf, assignId=0) # always assign to conformer 0 by default
    
    @classmethod
    def from_rdkit(cls, mol : Mol, positions : Optional[ndarray[Shape[N, 3], float]]=None) -> 'Primitive':
        '''Initialize a chemically-explicit Primitive from an RDKit Mol'''
        mol_sanitized = sanitize_mol(
            mol,
            sanitize_ops=SANITIZE_ALL,          # TODO: add option to specify sanitation
            aromaticity_model=AROMATICITY_MDL,  # TODO: add option to specify sanitation
            in_place=False, # don't modify original
        )
        
        primitive = cls(
            num_atoms=mol_sanitized.GetNumAtoms(),
            functionality=get_num_linkers(mol_sanitized), # TODO: generalize Ports to work EVEN without explicit chemistry
            shape=None,
            chemistry=mol_sanitized,
            smiles=Chem.MolToSmiles(
                mol,
                canonical=True,         # canonicalize to allow uniqueness check
                isomericSmiles=True,    # include stereo info
                allHsExplicit=True,     # include all Hs
                allBondsExplicit=True,  # show all bonds
                ignoreAtomMapNumbers=False, # preserve mapped atoms
                doRandom=False,         # make SMILES deterministic
            ),
            stereo_marker=None, # TODO: add perception of this
        )

        if (positions is None) and (mol_sanitized.GetNumConformers() > 0): # attempt to get positions from Mol directly, if no explicit positions as passed but a conformer is present 
            positions = mol_sanitized.GetConformer(0).GetPositions() # for now, will always assume active conformer is the first
        primitive.set_positions(positions) # harmonize positions between shape and chemistry

        return primitive
    
    @classmethod
    def from_SMILES(cls, smiles : str, positions : Optional[ndarray[Shape[N, 3], float]]=None) -> 'Primitive':
        '''Initialize a chemically-explicit Primitive from a SMILES string representation of the molecular graph'''
        return cls.from_rdkit(
            mol=Chem.MolFromSmiles(smiles, sanitize=False), # don't mangle molecule with default sanitization - read SMILES verbatim
            positions=positions,    
        )


@dataclass
class PrimitiveLexicon:
    '''Collection of primitives which form the basis of a'''
    primitives : dict[str, Primitive] = field(default_factory=dict) # priitives keyed by aliases
    
    # this will likely need **some** methods, but most of the sanitization ought to be the responsibility of the MID graph
    # this is more-or-less a glorified container for now
       