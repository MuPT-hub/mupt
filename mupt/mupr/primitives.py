'''Information classes for sets of polymer unit primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional
from dataclasses import dataclass, field
from enum import Enum # consider having a bitwise Enum to encode possible specification states of a primitive??

from rdkit import Chem

from .geometry.shapes import BoundedShape
from .chemistry.conformations import Conformer


@dataclass
class MolecularPrimitive:
    '''Fundamental building block which represents a grouping of some number of molecules'''
    # excluding naming for now as this may screw up comparison of otherwise identical primitives (maybe we'd want that down the line?)
    num_atoms     : Optional[int] = None # number of atoms
    functionality : Optional[int] = None # number of linker sites, which can connect to other primitives
    chemistry     : Optional[str] = None # a SMILES-like string which represents the internal chemistry of the primitive
    conformer     : Optional[Conformer] = None # the spatial coordinates of constituent atoms (if chemistry is specified)
    shape         : Optional[BoundedShape] = None # a rigid shape which approximates and abstracts the behavoir of the primitive as a unit
    # TODO: conformer could be made a BoundedShape through the use of a convex hull around the atom coords; this would further unify how primitives are treated
    
    # Note that, by default ALL fields are optional; this is to reflect the fact that use-cases and levels of info provided may vary
    # 
    # For example, one might object that functionality and number of atoms could be derived from the SMILES string and are therefore redundant;
    # However, in the case where no chemistry is explicitly provided, it's still perfectly valid to define numbers of atoms present
    # E.g. a coarse-grained sticker-and-spacer model
    #
    # As another example, a 0-functionality primitive is also totally legal (ex. as a complete small molecule in an admixture)
    # But comes with the obvious caveat that, in a network, it cannot be incorporated into a larger component

    def __post_init__(self) -> None:
        self._cleanup()
        
    def _cleanup(self) -> None:
        '''Perform some sanitization based on which inputs are provided and what those inputs are'''
        if self.chemistry is None:
            return # skip cleanup if no internal chemistry is provided
        
        mol = Chem.MolFromSmiles(self.chemistry, sanitize=False)
        # will need some sanitization here to guarantee that chemistry (Hs, bond orders, charges, etc) are fully-explicit
        # consider raising Exceptions, instead of quietly overriding?
        self.functionality = 0 # insert mechanism for counting linker sites here
        self.num_atoms = mol.GetNumAtoms() - self.functionality # linker sites are virtual ("*"-type) atoms which should not be counted as "real" atoms
        
        if self.conformer is not None:
            assert len(self.conformer) == self.num_atoms
            # intersection test maybe?
            if self.shape is not None:
                for coord in self.conformer:
                    assert(self.shape.contains(coord)) # this is maybe too stringent, but simplifies primitive-level intersections later on
        
    # would also be cool to have chemistry-free method of labelling ports (perhaps passing Sequence of Port-type objects in addition to chemistry?)
        
    def __hash__(self):
        raise NotImplemented # critical that this exists to allow comparison between primitives
    
    def __eq__(self, value):
        raise NotImplemented # chemical equality will likely be the hardest part to compare - SMILES canonicalization might alleviate this?

@dataclass
class PrimitiveLexicon:
    '''Collection of primitives which form the basis of a'''
    primitives : dict[str, MolecularPrimitive] = field(default_factory=dict) # priitives keyed by aliases
    
    # this will likely need **some** methods, but most of the sanitization ought to be the responsibility of the MID graph
    # this is more-or-less a glorified container for now
       