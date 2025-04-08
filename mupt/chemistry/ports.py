'''Definition of inter-atom bonding placeholders and how they interact'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

# want to think about how neutronium ("*~*") groups are considered for bonding before transferring from polymerist

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Atom, Bond, Mol


class MolPortError(Exception):
    '''Raised when port-related errors as encountered'''
    pass

def is_linker(rdatom : Atom) -> bool:
    '''Indicate whether an atom is a linker (intermonomer "*" type atom)'''
    return rdatom.GetAtomicNum() == 0

def get_num_linkers(rdmol : Mol) -> int:
    '''Count how many wild-type inter-molecule linker atoms are in a Mol'''
    return sum(
        is_linker(atom)
            for atom in rdmol.GetAtoms()
    )
    
@dataclass(frozen=True)
class Port:
    '''Class for encapsulating the components of a "port" bonding site (linker-bond-bridgehead)'''
    linker     : Atom
    bond       : Bond
    bridgehead : Atom

    # bondable_flavors : ClassVar[UnorderedRegistry] = field(default=UnorderedRegistry((0, 0))) # by default, only two (0, 0)-flavor ("unlabelled") ports are bondable

    @property
    def flavor(self) -> int:
        '''Return the flavor of the port'''
        return self.linker.GetIsotope()
    
    def __eq__(self, other : 'Port') -> bool:
        raise NotImplemented # criteria for bonding will be defined here