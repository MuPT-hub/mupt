Keeping stereocenters through coordinate building
=================================================

A MuPT system usually starts from coordinates that already have the right
chemistry. :func:`~mupt.interfaces.smiles.primitive_from_smiles` embeds each
residue from a SMILES string, and every ``@`` or ``@@`` in that string becomes
a real handedness in the embedded geometry. Coordinate builders then move the
atoms. A placement that only rotates and translates whole residues cannot
change a handedness. Builders that move atoms individually can. In all-atom DPD
(:class:`~mupt.builders.all_atom_dpd.AllAtomDPDBuilder`) the pair repulsion is
soft, so an atom can pass through a stereocenter and invert it. A later
classical force field cannot invert it back, and the simulation silently
describes a different molecule (`#116
<https://github.com/MuPT-hub/mupt/issues/116>`_).

The stereochemistry guard in :mod:`mupt.chemistry.stereo` detects this. It
reads the handedness of every stereocenter in the starting coordinates, and
reports any center whose handedness differs in a later set of coordinates.
It is a diagnostic on coordinates only. Primitives never store
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

The guard flattens the whole hierarchy below the Primitive it is given into
atoms and bonds, regardless of residues, segments or any other grouping. A
backbone carbon whose fourth neighbor belongs to the next residue is
therefore found like any other center.

Centers are found from the bond graph alone, so hydrogens must be explicit.
An atom with four neighbors is a stereocenter if no two of its neighbors are
equivalent under the symmetry of the molecule. This leaves out every CH2 and
CH3 group, where swapping two hydrogens changes nothing. Three kinds of
center are kept even though two of their neighbors look alike, because
inverting them still makes a different molecule:

- centers of meso compounds, such as both carbons of meso-tartaric acid;
- cis/trans ring centers, such as C1 and C4 of 1,4-dimethylcyclohexane;
- pseudoasymmetric centers, such as the middle backbone carbon of a symmetric
  tactic chain.

Pyramidal sulfur, phosphorus, selenium and arsenic with three neighbors
(sulfoxides, phosphines) are also centers. Three-coordinate nitrogen is not,
since amines invert on their own at room temperature. To check a specific
set of atoms, pass their indices as ``centers``.

Finding centers takes time linear in the number of atoms, so a melt of long
chains costs no more per atom than a small molecule. Each branch of the
molecule is hashed once, from its tips inward, and each ring system is hashed
as a single unit.


Handedness at residue junctions
-------------------------------

Residues embedded one at a time are not yet placed relative to one another.
For a center at a junction, the neighbor in the next residue may be anywhere.
:func:`~mupt.chemistry.stereo.capture_stereo` handles this through
Connectors. It takes each neighbor of a center to sit at the linker of the
center atom's Connector for that bond. Within a residue, the linker coincides
with the neighbor atom. Across residues, the linker is where the residue's
own template placed its bond partner, the ``*`` of the residue SMILES. A
junction center therefore gets the handedness that its residue template
defines, before any placement.

The check always uses atom positions. Builders that write new atom positions
without moving Connectors, as AA-DPD does, leave stale linkers behind. To
capture a reference from such coordinates, pass
``use_connector_positions=False``.


How handedness is measured
--------------------------

For a center with neighbors a, b, c and d (in atom index order), the guard
computes the signed volume ``det(a - d, b - d, c - d)`` and divides it by the
three edge lengths, giving a value between -1 and 1 that does not depend on
bond lengths. A reflection flips its sign. Rotations and translations do not
change it. The sign is compared between two geometries of the same atoms, so
it needs no R/S assignment. For a center with a lone pair, the center itself
stands in for d.
