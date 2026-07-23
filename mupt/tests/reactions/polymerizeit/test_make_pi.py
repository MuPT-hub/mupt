'''Unit tests for the make_pi PolymerizeIt! orchestrator'''

__author__ = 'Salman Bin Kashif'
__email__ = 'salmanbinkashif@gmail.com'

import textwrap
from unittest.mock import patch, call, Mock

import pytest

from mupt.reactions.polymerizeit import react_pi, init_pi
from mupt.reactions.polymerizeit.make_pi import make_pi


# Minimal PolymerizeIt! input (mupt-examples schema): 2 monomers, 1 reaction, 1 system.
# Enough to assert make_pi's per-reaction / per-system orchestration. react_pi and init_pi are mocked,
# so the SMILES need only be parseable strings and no chemistry actually runs.
MINIMAL_INPUTS_YAML = textwrap.dedent('''\
    dirname: test_poly
    monomers:
      - {name: A, smi: 'CCO'}
      - {name: B, smi: 'CCN'}
    reactions:
      rxn1:
        reactants: [A, B]
        react_template: ['[O:1]', '[N:2]']
        react_idx: [0, 0]
        products: [DIM]
        prod_template: ['[O:1][N:2]']
        prod_idx: [0]
    reaction_engine:
      name: polymerizeit
      inputs: {num_iterations: 1, protocol: default}
    system_parameters:
      - name: sysA
        temperature: 300
        density: 0.8
        total_atoms: 1000
        monomer_ratios: {A: 1.0, B: 1.0}
    ''')


@pytest.fixture(scope='function')
def minimal_inputs_file(tmp_path):
    '''Write a minimal PolymerizeIt! inputs.yaml (make_pi reads a file path) and return it.'''
    path = tmp_path / 'inputs.yaml'
    path.write_text(MINIMAL_INPUTS_YAML)
    return path


def test_make_pi_orchestrates_reactions_then_systems(minimal_inputs_file):
    '''make_pi seeds working lists, drives react_pi per reaction then init_pi per system, and returns inputs.'''
    manager = Mock()  # records cross-mock call ordering
    with (
        patch.object(react_pi, 'react_molecules_to_product') as m_react,
        patch.object(react_pi, 'identify_reactive_sites') as m_ident,
        patch.object(react_pi, 'map_product_atoms_to_reactants') as m_map,
        patch.object(init_pi, 'generate_config_file') as m_gen,
        patch.object(init_pi, 'fill_config') as m_fill,
    ):
        for name, mock in [('react', m_react), ('ident', m_ident), ('map', m_map),
                           ('gen', m_gen), ('fill', m_fill)]:
            manager.attach_mock(mock, name)
        inputs = make_pi(inputs_file=str(minimal_inputs_file))

    # returns the mutated inputs dict
    assert isinstance(inputs, dict)

    # working lists seeded from monomers (deep copy, not aliased)
    assert inputs['molecules'] == inputs['monomers']
    assert inputs['molecules'] is not inputs['monomers']
    assert inputs['repeat_units'] == []
    assert inputs['byproducts'] == []

    # react_pi entry point called once per reaction, with (inputs, reaction_key)
    assert m_react.call_args == call(inputs, 'rxn1')

    # exact call sequence for 1 reaction + 1 system: all react_pi steps, then init_pi steps
    sequence = [recorded[0] for recorded in manager.mock_calls]
    assert sequence == ['react', 'ident', 'map', 'gen', 'fill']
