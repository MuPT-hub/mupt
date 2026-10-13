Keeping stereocenters through coordinate building
=================================================

A MuPT system usually starts from coordinates that already have the right
chemistry. :func:`~mupt.interfaces.smiles.primitive_from_smiles` embeds each
residue from a SMILES string, and every ``@`` or ``@@`` in that string becomes
a real handedness in the embedded geometry. Coordinate builders then move the
atoms. Rigid transformations don't change stereochemistry. Physics-based builders may move individual atoms, potentially inverting stereocenters. In all-atom DPD
(:class:`~mupt.builders.all_atom_dpd.AllAtomDPDBuilder`) the pair repulsion is
soft, so an atom can pass through a stereocenter and invert it. A later
classical force field cannot invert it back, and the simulation silently
describes a different molecule (`#116
<https://github.com/MuPT-hub/mupt/issues/116>`_).

The stereochemistry guard in :mod:`mupt.chemistry.stereo` detects this. It
reads the handedness of every stereocenter in the starting coordinates, and
reports any center whose handedness differs in a later set of coordinates.
It is a diagnostic on coordinates only. Primitives do not store
stereocenters or labels.


Using the guard
---------------

Capture a reference before building, and check the result afterwards:

.. code-block:: python

    from mupt.chemistry.stereo import capture_stereo, assert_stereo_preserved

    reference = capture_stereo(root)        # starting coordinates
    result = builder.build(root)            # any builder; AA-DPD mutates root
    assert_stereo_preserved(root, reference, box_lengths=result.box_lengths_a)

:func:`~mupt.chemistry.stereo.check_stereo` returns a
:class:`~mupt.chemistry.stereo.StereoReport` instead of raising. Every
recorded center gets one status: ``preserved``, ``inverted``,
``near_planar`` (flattened, so its handedness is ambiguous) or ``nonfinite``.
Both functions also accept a plain ``(N, 3)`` array of positions in place of
a Primitive, for checking coordinates that have not been written back yet.
Pass ``box_lengths`` when the coordinates are wrapped into a periodic box.


What counts as a stereocenter
-----------------------------

The guard operates on a flat slice of the hierarchy: every atom and bond below
the Primitive it is given, regardless of residues or other grouping. Every
atom with four bonded neighbors is checked, with no exceptions. That includes
CH2 and CH3 carbons, whose inversion would give back the same molecule, so the
contract is simply that no tetrahedral center changes handedness. Hydrogens
must be explicit. To check only some atoms, pass their indices as ``centers``.


Handedness at residue junctions
-------------------------------

A stereocenter at a residue junction has one neighbor in the adjacent residue.
Before residues are placed, that neighbor can be anywhere, so
:func:`~mupt.chemistry.stereo.capture_stereo` uses the linker of the Connector
that forms the bond instead. The linker sits where the residue SMILES put its
``*``, so the reference is the handedness the residue template defines. Inside
a residue, the linker and the neighbor atom are in the same place.

The check always uses atom positions. Some builders, including AA-DPD, move
atoms but not Connectors, so capture the reference before building. To capture
from a builder's output, pass ``use_connector_positions=False``.


How handedness is measured
--------------------------

For a center with neighbors a, b, c and d (in atom index order), the guard
computes the signed volume ``det(a - d, b - d, c - d)`` and divides it by the
three edge lengths, giving a value between -1 and 1 that does not depend on
bond lengths. A reflection flips its sign. Rotations and translations do not
change it. The sign is compared between two geometries of the same atoms, so we don't need to assign R/S.
