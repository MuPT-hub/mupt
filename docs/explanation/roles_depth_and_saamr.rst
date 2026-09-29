Meaning Across Scales: Roles, Depth, and SAAMR
===============================================

.. DRAFT (Joe): this page is a first draft generated for review. Comments
   starting with "DRAFT" are open questions and are not rendered. Remove
   them before merging.

A MuPT representation is a tree of :class:`~mupt.mupr.primitives.Primitive`
objects. The tree can be as deep or as shallow as a problem needs: a
polymer melt might be a universe of chains, each chain a sequence of
repeat units, each repeat unit a set of atoms; a block copolymer might add
a level for blocks; a coarse-grained model might stop at beads. MuPT does
not assign any fixed meaning to a given depth in the tree. This is what
allows the same data structure to describe a system at many resolutions.

Most of the tools MuPT hands systems off to do not work this way.
MDAnalysis organizes a system into segments, residues, and atoms. RDKit
works one molecule at a time. The PDB format expects chains, residue
numbers, and atoms. Each of these has a fixed set of levels, and each level
has a specific meaning.

*Roles* are how MuPT bridges the two. A role is an explicit label on a
Primitive that says what that Primitive *means* to the outside world,
independent of where it sits in the tree.


Why depth is not enough
-----------------------

The simplest way to map a MuPT tree onto an external toolkit is to assume
a layout: the root is the system, its children are molecules, their
children are residues, and their children are atoms. Early versions of the
MDAnalysis exporter did exactly this, walking the tree with nested loops
that assumed every atom sat at depth 3.

That assumption breaks as soon as a tree has a different shape. Adding a
single grouping level (for example, collecting chains into a "domain", or
residues into blocks) shifts every atom one level deeper, and a
depth-based exporter will silently mislabel residues as segments and atoms
as residues. Worse, different parts of the code could disagree about
ordering, producing topologies whose residues were scrambled relative to
their coordinates (see `issue #19 <https://github.com/MuPT-hub/mupt/issues/19>`_).

The underlying problem is that depth is *structural* information, while
"this is a residue" is *semantic* information. Roles make the semantic
information explicit, so exporters never have to infer it from depth.


What a role is
--------------

Every Primitive has a :attr:`~mupt.mupr.primitives.Primitive.role`
attribute, whose value is a member of the
:class:`~mupt.roles.PrimitiveRole` enum:

``UNIVERSE``
    The root container of the whole system.

``SEGMENT``
    A covalently self-contained entity, such as a polymer chain, a small
    molecule, or an ion. Nothing is bonded *between* segments.

``RESIDUE``
    A repeating sub-unit of a segment, such as a monomer or an amino acid.

``PARTICLE``
    The smallest exported unit. In an all-atom representation this is an
    atom.

``UNASSIGNED``
    The default. An unassigned Primitive has no meaning to an exporter and
    is treated as a *transparent* grouping node: exporters look through it
    to the role-bearing Primitives beneath.

.. DRAFT: the PrimitiveRole docstring says PARTICLE may also be a "bead in
   CG", but every current export path requires PARTICLE leaves to have an
   element. Say CG PARTICLEs are planned (#109), or drop the mention?

A few properties of roles are worth knowing:

- **Roles are opt-in.** MuPT never assigns roles on its own. Either you set
  them when building a Primitive (``Primitive(..., role=PrimitiveRole.RESIDUE)``),
  set them afterwards (``prim.role = PrimitiveRole.RESIDUE``), or call a
  helper such as :func:`~mupt.roles.assign_SAAMR_roles`.
- **Roles are preserved by copying.** ``Primitive.copy()`` carries roles through to the copied tree, so a residue template tagged
  once stays tagged when it is copied into many chains.
- **Roles do not affect identity.** Two Primitives that differ only in
  their roles compare equal and have the same canonical form. A role
  describes how a Primitive should be *presented* to other tools, not what
  it *is* chemically.

.. DRAFT: confirm with Tim that excluding role from canonical_form/__eq__ is
   intended, and whether role is expected to stay a core Primitive attribute
   after #56 (it moved to metadata and back during #50 review).


SAAMR: one system of roles
--------------------------

The four non-default roles are not arbitrary. Together they form one
particular convention, the **Standard All-Atom Molecular Representation
(SAAMR)**, which mirrors the universe / segment / residue / atom hierarchy
used by MDAnalysis and, more loosely, by most biomolecular and polymer
file formats.

.. list-table:: SAAMR roles and their counterparts
   :header-rows: 1
   :widths: 15 25 20 20 20

   * - Role
     - Meaning
     - MDAnalysis
     - RDKit
     - PDB
   * - ``UNIVERSE``
     - the whole system
     - ``Universe``
     - (set of Mols)
     - the file
   * - ``SEGMENT``
     - one covalent entity
     - segment
     - one ``Mol``
     - chain (see note below)
   * - ``RESIDUE``
     - repeat unit / monomer
     - residue
     - atom properties
     - residue
   * - ``PARTICLE``
     - atom
     - atom
     - atom
     - atom

SAAMR is *a* system of roles, not the only possible one. It is the system
that MuPT's current all-atom exporters and builders understand, which is
why it gets a name. Other conventions (for example, coarse-grained
hierarchies, or labels for functional groups, Kuhn segments, or
sticker/spacer units) are natural extensions of the same idea; see
`Beyond SAAMR`_.


Roles versus depth in practice
------------------------------

The simplest SAAMR tree is one where the roles and the depths line up
exactly: universe at depth 0, segments at depth 1, residues at depth 2,
and atoms at depth 3. MuPT calls this *strict SAAMR depth*, and for trees
with this shape, :func:`~mupt.roles.assign_SAAMR_roles` will label every
level for you:

.. code-block:: python

    from mupt.chemistry import ELEMENTS
    from mupt.mupr.primitives import Primitive
    from mupt.mupr.properties import has_strict_SAAMR_depth
    from mupt.roles import assign_SAAMR_roles, has_SAAMR_roles

    universe = Primitive(label="universe")
    chain = Primitive(label="chain")
    unit = Primitive(label="HEL")
    atom = Primitive(label="He", element=ELEMENTS[2])

    universe.attach_child(chain)
    chain.attach_child(unit)
    unit.attach_child(atom)

    has_strict_SAAMR_depth(universe)  # True: every leaf is an atom at depth 3
    has_SAAMR_roles(universe)         # False: nothing is labelled yet

    assign_SAAMR_roles(universe)
    has_SAAMR_roles(universe)         # True

Real systems are often not this tidy. Suppose the chain is grouped under
an extra "domain" node. The atom now sits at depth 4, so the tree no
longer has strict SAAMR depth and ``assign_SAAMR_roles`` will refuse to
guess, raising a ``ValueError``. The roles can still be set by hand, and
the domain node is simply left ``UNASSIGNED``:

.. code-block:: python

    from mupt.interfaces.mdanalysis import primitive_to_mdanalysis
    from mupt.roles import PrimitiveRole

    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    domain = Primitive(label="domain")  # UNASSIGNED: transparent
    chain = Primitive(label="chain", role=PrimitiveRole.SEGMENT)
    unit = Primitive(label="HEL", role=PrimitiveRole.RESIDUE)
    atom = Primitive(label="He", element=ELEMENTS[2], role=PrimitiveRole.PARTICLE)

    universe.attach_child(domain)
    domain.attach_child(chain)
    chain.attach_child(unit)
    unit.attach_child(atom)

    has_strict_SAAMR_depth(universe)  # False: the atom is at depth 4
    has_SAAMR_roles(universe)         # True

    u = primitive_to_mdanalysis(universe, resname_map={})
    # 1 segment, 1 residue ("HEL"), 1 atom -- the domain level is looked through

This is the central idea of the page: **depth describes how a tree is
organized; roles describe what its parts mean.** The two coincide in a
strict SAAMR tree, but exporters only ever rely on the roles.

It helps to keep the three checks straight:

:func:`~mupt.mupr.properties.has_strict_SAAMR_depth`
    A *structural* check: are all leaves atoms at depth exactly 3? This is
    the precondition for :func:`~mupt.roles.assign_SAAMR_roles`, and
    nothing else depends on it.

:func:`~mupt.roles.has_SAAMR_roles`
    A quick *presence* check: does at least one Primitive carry each of the
    four SAAMR roles? It does not check how those roles are arranged.

The exporters' own validation
    The full contract, described in the next section. Every SAAMR-aware
    exporter checks it before writing anything and raises a ``ValueError``
    explaining what is wrong.

.. DRAFT: has_SAAMR_roles is presence-only and unused in production. Worth
   recommending that the exporter validator become public (it lives in the
   private mupt.interfaces._shared.topology)?

.. DRAFT: assign_SAAMR_roles overwrites any roles already set. Mention as a
   caveat, or treat as a bug?


Rules for a well-formed SAAMR tree
----------------------------------

The rules the exporters actually enforce are role-based, not depth-based.
A tree is exportable as SAAMR when:

1. The root has the ``UNIVERSE`` role.
2. Every leaf is a ``PARTICLE`` with an element, and only leaves are
   ``PARTICLE``\ s.
3. Every ``PARTICLE`` is contained (at any depth) in a ``RESIDUE``, and
   every ``RESIDUE`` in a ``SEGMENT``.
4. ``SEGMENT``\ s do not contain other ``SEGMENT``\ s, and ``RESIDUE``\ s
   do not contain other ``RESIDUE``\ s.
5. No ``SEGMENT`` is empty of residues, and no ``RESIDUE`` is empty of
   particles.
6. Bonds are owned at or below the ``SEGMENT`` level. A Primitive above
   any segment may not own internal connections, because a bond there
   would join two segments that are supposed to be covalently separate.

Anything else is allowed. In particular, ``UNASSIGNED`` Primitives may
appear anywhere between these levels: between the universe and its
segments, between a segment and its residues, or between a residue and
its atoms.

Rule 6 is what gives ``SEGMENT`` its meaning. Everything covalently bonded
together belongs to one segment, which is why a segment maps cleanly onto
a single RDKit ``Mol`` or a single MDAnalysis segment.


How roles are used
------------------

MDAnalysis export
    :func:`~mupt.interfaces.mdanalysis.exporters.primitive_to_mdanalysis`
    turns each ``SEGMENT`` into an MDAnalysis segment, each ``RESIDUE``
    into a residue, and each ``PARTICLE`` into an atom, in a deterministic
    order. Residue names come from the residue's ``"residue_name"``
    metadata if present, then from ``resname_map``, then from its label,
    and must be three characters long.

RDKit export
    :func:`~mupt.interfaces.rdkit.exporters.primitive_to_rdkit_mols`
    yields one RDKit ``Mol`` per ``SEGMENT``. Residue and particle
    membership is recorded on each RDKit atom as properties
    (``mupt_segment_index``, ``mupt_residue_index``, and so on), so the
    SAAMR hierarchy can be recovered from the molecule.

    .. note::

       The PDB chain IDs and residue numbers written on these molecules
       are placeholders to satisfy the PDB format's limits (residue
       numbers roll over into the next chain letter after 9999). They do
       **not** encode segment identity; use the ``mupt_*`` atom properties
       for that.

SDF files
    MuPT's SDF writer is built on the RDKit exporter. The reader rebuilds a
    strict four-level UNIVERSE / SEGMENT / RESIDUE / PARTICLE tree, so
    transparent ``UNASSIGNED`` levels are not preserved through a round
    trip.

    .. DRAFT: SDF currently lives in mupt.temporary, which is not in the
       rendered API. Keep this entry or drop it until SDF is promoted?

All-atom DPD initialization
    :class:`~mupt.builders.all_atom_dpd.AllAtomDPDBuilder` uses the same
    role rules to find the chains, residues, and atoms it needs to place
    and relax. The geometric placement step itself is role-agnostic; the
    builder is the layer that reads roles and hands placement a
    chain-of-residues view of each segment.


Beyond SAAMR
------------

SAAMR is the first role system MuPT supports, not the last. Exporters are
built around interchangeable *strategies* (for example, an all-atom
strategy for MDAnalysis), so a different role convention, such as a
coarse-grained hierarchy where ``PARTICLE``\ s are beads, can be supported
by adding a strategy rather than rewriting the exporter.

What is fixed is the principle: meaning is attached to Primitives
explicitly, and tools that need a fixed hierarchy read that meaning
instead of guessing it from the shape of the tree.

.. DRAFT: forward-looking. Decide with the team whether future role
   vocabularies extend PrimitiveRole or get separate enums, and whether to
   reference #56 / #109 here.


See also
--------

- :mod:`mupt.roles` and :mod:`mupt.mupr.properties` in the API reference
- :doc:`molecular_representation`, for the Primitive tree itself
- :doc:`connectors`, for how bonds between Primitives are represented
