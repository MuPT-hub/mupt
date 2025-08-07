'''Structure definitions specific to atom-containing Primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Union
type Smileslike = str

from rdkit.Chem.rdchem import Atom
from rdkit.Chem.rdmolfiles import AtomFromSmiles, AtomFromSmarts

from .structure import Structure


class AtomicStructure(Structure[Atom]):
    '''A Structure representing a single atom from the periodic table'''
    def __init__(self, atom : Union[Atom, Smileslike]) -> None:
        if isinstance(atom, str): # attempt SMILES/SMARTS upconversion
            atom = AtomFromSmiles(atom)
            if atom is None: # fallback to more general SMARTS if string pattern is not recognized as SMILES
                atom = AtomFromSmarts(atom)

        if not isinstance(atom, Atom): # DEVNOTE: implicitly, catches None returned when str input is not a valid SMARTS either
            raise TypeError(f'Primitive structure must be an Atom, not {type(atom)}')
        
        self.atom = atom
    
    @property
    def num_atoms(self) -> int:
        return 1
    
    @property
    def is_composite(self) -> bool:
        return False
    
    def _get_components(self) -> tuple:
        return tuple() # no further internal components
    
    def canonical_form(self) -> str:
        symbol : str = self.atom.GetSymbol()
        if self.atom.GetIsAromatic():
            symbol = symbol.lower() # for now, encode aromaticity thru case; see how well (or poorly) this generalizes later

        return symbol # TODO: make this more expressive to capture stereo, aromaticity etc
