'''Unit tests for chemical valence validation'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest
from mupt.chemistry.core import valence_allowed


@pytest.mark.parametrize(
    'atomic_num,charge,valence,expected_allowed_valence',
    [
        # neutral atoms
        (1, 0, 1, True),
        (1, 0, 0, False),
        (2, 0, 0, True),
        (2, 0, 1, False), # bound helium
        (6, 0, 4, True),  # "standard" 4-functional carbon
        (6, 0, 5, False), # pentavalent carbon
        (7, 0, 3, True),  # freebase nitrogen
        (8, 0, 2, True),  # carbonxyl oxygen
        ## sulfur - many accessible valence states thru d-block electrons
        (16, 0, 2, True),
        (16, 0, 4, True),
        (16, 0, 6, True),
        # ions
        (1, 1, 0, True),   # free proton
        (7, 1, 4, True),   # ammonium nitrogen 
        (8, -1, 1, True),  # deprotonated oxygen
        (20, 2, 0, True),  # free calcium ion
        (20, 2, 1, False), # singly-bound calcium ion
        # linker atoms - to test that valence checks are insensitive (as they should be)
        (0, 0, 0, True),
        (0, 0, 42, True),
        (0, 1, 0, True),
        (0, 1, 42, True),
        (0, -1, 0, True),
        (0, -1, 42, True),
    ]
)
def test_valence_allowed(atomic_num : int, charge : int, valence : int, expected_allowed_valence : bool) -> None:
    '''Test that chemical valence validity checker makes chemical sense'''
    assert valence_allowed(atomic_num, charge, valence) == expected_allowed_valence