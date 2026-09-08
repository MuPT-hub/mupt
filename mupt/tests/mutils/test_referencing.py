'''Unit tests for object referencing'''

from dataclasses import dataclass

import pytest
from mupt.mutils.referencing import Addressable


def test_has_address() -> None:
    '''Test that Addressable objects indeed implement the address they claim to'''
    ...

def test_address_REgistration() -> None:
    '''Test that newly-minted objects are also registered by their address in the classwide registry'''
    ...

def test_weak_address_refs() -> None:
    '''Test that records of objects in classwide registry automatically vanish when object is garbage collected'''
    ...

def test_object_registries_distinct() -> None:
    '''Test that distinct subtypes of Addressable do not share their classwide object registries'''
    ...