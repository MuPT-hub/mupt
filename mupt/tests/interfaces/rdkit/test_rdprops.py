'''Unit tests for RDKit property assignment and lookup'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest

from typing import Any
from itertools import product as cartesian

from rdkit import Chem
from mupt.interfaces.rdkit.rdprops import (
    RDObj,
    isrdobj,
    assign_property_to_rdobj,
    copy_rdobj_props,
    RDPROP_GETTERS,
    RDPROP_SETTERS,
)


TEST_MOL = Chem.MolFromSmiles('O=C=O')  # carbon dioxide
TEST_RDOBJS : tuple[RDObj, ...] = (
    TEST_MOL.GetAtomWithIdx(0), # Atom
    TEST_MOL.GetBondWithIdx(0), # Bond
    TEST_MOL,                   # Mol
    Chem.RWMol(TEST_MOL)        # RWMol
)

TEST_PROP_VALUES : tuple[tuple[Any, str], ...] = (
    ('a string'  , 'GetProp'),
    (37          , 'GetIntProp'),
    (0.5772156649, 'GetDoubleProp'),
    (True        , 'GetBoolProp'),
    ([1, 2, 3]   , 'GetProp'), # should be coerced to string
    (None        , 'GetProp'), # should also be coerced to string
)
@pytest.mark.parametrize(
    "rdobj,prop_value,expected_getter_method_name",
    [
        (rdobj, *prop_inputs)
            for (rdobj, prop_inputs) in cartesian(TEST_RDOBJS, TEST_PROP_VALUES)
    ],
)
def test_rdprop_assignment(rdobj : RDObj, prop_value : Any, expected_getter_method_name : str) -> None:
    '''Test type-safe assignment of properties to RDKit objects'''
    prop_key : str = f'test_{type(prop_value).__name__}_prop'
    assign_property_to_rdobj(rdobj, prop_key, prop_value, preserve_type=True)
    
    getter = getattr(rdobj, expected_getter_method_name)
    _ = getter(prop_key) # NOTE: no need for assert; will raise Exception if value and type targetted by getter disagree
