'''For converting objects and types between different chemical representation formats'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Union

from rdkit.Chem.rdchem import Atom
from periodictable.core import Element, Ion, Isotope, isatom

from .core import ELEMENTS, ElementLike


def element_to_rdkit_atom(element : ElementLike) -> Atom:
    '''Convert a periodictable ElementLike instance to an RDKit Atom instance'''
    if not isatom(element):
        raise ValueError(f"Expected an ElementLike instance, got object of type {type(element).__name__}")
    
    atom = Atom(element.number)
    atom.SetNoImplicit(True) # prevent RDKit from adding on any Hs where we don't expect
    atom.SetFormalCharge(element.charge)
    if hasattr(element, 'isotope'): # covers case for charged isotopes, with are Ion (not Isotope) instances
        atom.SetIsotope(element.isotope)
        
    return atom

def rdkit_atom_to_element(atom : Atom) -> ElementLike:
    '''Convert an RDKit Atom instance to a periodictable ElementLike instance'''
    if not isinstance(atom, Atom):
        raise ValueError(f"Expected an RDKit Atom instance, got object of type {type(atom).__name__}")
    
    atom_is_linker : bool = (atom.GetAtomicNum() == 0) # DEV: am well-aware is_linker(atom : Atom) exists in the RDKit interface ("I wrote the damn bill"), but that import here would be circular
    elem : ElementLike = ELEMENTS[atom.GetAtomicNum()]
    if (mass_number := atom.GetIsotope()) != 0:
         # bypass isotope validity check ONLY for linker atoms (not actually neutrons, like periodictable seems to think they are!)
        elem = Isotope(elem, mass_number) if atom_is_linker else elem[mass_number]
    
    # fetch Ion instance - NOTE: order here is deliberate; can't fetch Isotope of Ion, but CAN fetch Ion of Isotope
    if (charge := atom.GetFormalCharge()) != 0:
        elem = Ion(elem, charge) if atom_is_linker else elem.ion[charge] 
    
    return elem

def flexible_elementlike(elem : Union[int, str, Atom, ElementLike]) -> ElementLike:
    '''Coerce inputs with a range of input types into ElementLike instance'''
    if isatom(elem):
        return elem
    elif isinstance(elem, int):
        return ELEMENTS[elem]
    elif isinstance(elem, str):
        return ELEMENTS.symbol(elem)
    elif isinstance(elem, Atom):
        return rdkit_atom_to_element(elem)
    else:
        raise TypeError(f'Cannot interpret object of type {type(elem).__name__} as ElementLike')