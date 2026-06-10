All-Atom DPD Melt Initialization
================================

The :mod:`mupt.builders.all_atom_dpd` module builds dense all-atom polymer
coordinates from a SAAMR MuPT hierarchy. It is intended as an initialization
method for downstream molecular dynamics, not as a production DPD model.

What The Builder Guarantees
--------------------------

``AllAtomDPDBuilder`` mutates atom ``PARTICLE`` leaf coordinates in place and
then recomputes parent ``PointCloud`` shapes. The builder also records a cubic
unit cell in ``root.metadata["unit_cell_parameters"]`` and a JSON-like summary in
``root.metadata["all_atom_dpd_summary"]``.

The HOOMD DPD simulation is used only to remove severe all-atom overlaps while
maintaining approximate bonded geometry. The DPD conservative and dissipative
parameters are heuristic and should not be interpreted as physically calibrated
DPD dynamics.

Recommended MD Handoff
----------------------

After AA-DPD placement, validate and equilibrate the coordinates in an atomistic
MD engine before production analysis:

1. Export the updated atom coordinates to the target MD representation.
2. Build an atomistic force-field system with explicit hydrogens and periodic
   box vectors from the AA-DPD result.
3. Assign production-quality partial charges. For OpenFF workflows, prefer the
   NAGL/AshGC model ``openff-gnn-am1bcc-1.0.0.pt`` over debug-only zero or
   formal charges.
4. Run unconstrained energy minimization and require finite energies.
5. Run short NVT dynamics with regular MD settings, for example a ``2 fs``
   timestep and ``1 / ps`` Langevin friction when hydrogen-bond constraints are
   present.
6. Run NPT equilibration at the intended temperature and pressure until density,
   volume, potential energy, and kinetic energy are bounded and stationary.
7. Use the equilibrated NPT density for any subsequent NVT production runs when
   cleaner structural or dynamical statistics are needed.

Do not use the sign of the total potential energy as a stability criterion. The
absolute zero of molecular-mechanics potential energy is force-field-dependent,
and bonded or torsional terms can make a stable system's total potential energy
positive. Prefer finite bounded energies, stable density/volume, and structural
observables.

Polyethylene Melt Validation
----------------------------

For polyethylene melt validation, run above the crystalline melting range. A
practical starting point is ``450 K`` and ``1 atm`` with an initial density near
``0.77 g / cm^3``. Literature PE melt densities are roughly ``0.76-0.78 g / cm^3``
near ``450 K`` and ``0.73-0.75 g / cm^3`` near ``500 K``, although the exact value
depends on force field, chain length, equilibration time, and finite-size effects.

The manual validation harness can be run from the repository root, for example:

.. code-block:: bash

   python devtools/scripts/validate_all_atom_dpd_pe_melt.py \
     --n-chains 20 \
     --chain-len 30 \
     --density-g-cm3 0.77 \
     --md-temperature-k 450 \
     --md-steps 100000 \
     --md-report-interval 25000 \
     --npt-steps 500000

This performs AA-DPD placement, OpenMM minimization, ``200 ps`` of NVT, and
``1 ns`` of NPT with ``2 fs`` timesteps. Treat this as a validation and
equilibration check, not as production sampling.

Physical Quality Checks
-----------------------

Density and finite energy are necessary but not sufficient for a realistic melt.
For stronger validation, compare against polymer-melt benchmarks using:

* NPT equilibrium density or specific volume.
* Radius of gyration and end-to-end distance.
* Backbone trans/gauche torsion populations.
* Intermolecular radial distribution functions.
* Mean-squared internal distances along the chain.
* Multiple independent random seeds.
* Larger systems and longer chains, for example C44-C100-like chains for
  unentangled polyethylene melt benchmarks.

Key Limitations
---------------

The AA-DPD builder initializes dense coordinates; it does not prove that the
resulting melt is equilibrated. Polymer conformational relaxation can require
longer simulations than the smoke tests used in continuous integration or manual
developer validation. Force-field choice also matters: an all-atom OpenFF model
may not reproduce polyethylene melt density as accurately as a force field tuned
specifically for polyethylene or n-alkane melts.
