'''Test translation between RDKit Atom and periodictable ElementLike instances'''

import pytest 

from rdkit.Chem.rdchem import Atom
from periodictable.core import Element, Ion, Isotope
from mupt.chemistry.core import (
    ELEMENTS,
    ElementLike,
    isatom,
)
from mupt.chemistry.conversion import (
    rdkit_atom_to_element,
    element_to_rdkit_atom,
    flexible_elementlike,
)

def compile_element_to_atom_params() -> dict[ElementLike, Atom]:
    '''Compile test examples for element to atom conversion tests (since parameterized pytest fixtures still aren't a thing)'''
    test_examples : dict[ElementLike, Atom] = {}
    
    # "pure" atoms
    test_examples[ELEMENTS[0]] = Atom(0) # n (linker/wildcard/"*")
    test_examples[ELEMENTS[1]] = Atom(1) # H
    test_examples[ELEMENTS[6]] = Atom(6) # C
    test_examples[ELEMENTS[26]] = Atom(26) # Fe
    
    # ions
    ammonium_nitrogen = Atom(7)
    ammonium_nitrogen.SetFormalCharge(1)
    test_examples[ELEMENTS[7].ion[1]] = ammonium_nitrogen
    
    chloride = Atom(17)
    chloride.SetFormalCharge(-1)
    test_examples[ELEMENTS[17].ion[-1]] = chloride
    
    phosphide = Atom(15)
    phosphide.SetFormalCharge(-3)
    test_examples[ELEMENTS[15].ion[-3]] = phosphide
    
    hexavalent_chromium = Atom(24)
    hexavalent_chromium.SetFormalCharge(6)
    test_examples[ELEMENTS[24].ion[6]] = hexavalent_chromium
    
    # isotopes
    labelled_linker = Atom(0)
    labelled_linker.SetIsotope(34) # test that arbitrary isotope can be applied to linker atoms
    test_examples[Isotope(ELEMENTS[0], 34)] = labelled_linker
    
    deuterium = Atom(1)
    deuterium.SetIsotope(2)
    test_examples[ELEMENTS.symbol('D')] = deuterium
    
    carbon_13 = Atom(6)
    carbon_13.SetIsotope(13)
    test_examples[ELEMENTS[6][13]] = carbon_13
    
    doubly_magic_lead = Atom(82)
    doubly_magic_lead.SetIsotope(208)
    test_examples[ELEMENTS[82][208]] = doubly_magic_lead
    
    fissile_uranium = Atom(92)
    fissile_uranium.SetIsotope(235)
    test_examples[ELEMENTS[92][235]] = fissile_uranium
    
    # charged isotopes
    linker_anion = Atom(0)
    linker_anion.SetIsotope(42)
    linker_anion.SetFormalCharge(-1) # DEV: not sure why you'd ever want this, but it IS technically supported
    test_examples[Ion(Isotope(ELEMENTS[0], 42), -1)] = linker_anion
    
    tritium_anion = Atom(1)
    tritium_anion.SetIsotope(3)
    tritium_anion.SetFormalCharge(-1)
    test_examples[ELEMENTS.symbol('T').ion[-1]] = tritium_anion
    
    carbon_13_cation = Atom(6)
    carbon_13_cation.SetIsotope(13)
    carbon_13_cation.SetFormalCharge(1)
    test_examples[ELEMENTS[6][13].ion[1]] = carbon_13_cation 
    
    return test_examples
    
def compile_atom_to_element_params() -> dict[Atom, ElementLike]:
    '''Compile test examples for atom to element conversion tests'''
    return {
        atom : elem
            for elem, atom in compile_element_to_atom_params().items()
    }
    
@pytest.mark.parametrize(
    'element, atom_expected',
    compile_element_to_atom_params().items()
)
def test_element_to_rdkit_atom(element : ElementLike, atom_expected : Atom) -> None:
    '''Test conversion from ElementLike to RDKit Atom instances'''
    atom_actual = element_to_rdkit_atom(element)
    
    assert (
        atom_actual.GetAtomicNum() == atom_expected.GetAtomicNum()
        and atom_actual.GetIsotope() == atom_expected.GetIsotope()
        and atom_actual.GetFormalCharge() == atom_expected.GetFormalCharge()
    )
    
@pytest.mark.parametrize(
    'atom, element_expected',
    compile_atom_to_element_params().items()
)
def test_rdkit_atom_to_element(atom : Atom, element_expected : ElementLike) -> None:
    '''Test conversion from RDKit Atom to ElementLike instances'''
    element_actual = rdkit_atom_to_element(atom)
    
    assert (
        type(element_actual) is type(element_expected) # need additional check to see if right on of {Element, Ion, Isotope} was returned
        and element_actual.number == element_expected.number
        and element_actual.charge == element_expected.charge
    )
    if hasattr(element_expected, 'isotope'):
        assert element_actual.isotope == element_expected.isotope
    