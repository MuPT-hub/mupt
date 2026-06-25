"""All-atom DPD coordinate builder for SAAMR-compliant Primitive hierarchies.

The builder uses OpenFF labels to construct bonded restraints and heuristic DPD
repulsions for dense coordinate initialization. The HOOMD simulation is meant to 
produce finite all-atom melt coordinates suitable for downstream minimization in 
an MD engine.

Particle treatment
------------------
This initializer currently treats every atom in the input hierarchy, including
explicit hydrogens, as a DPD particle. A future optimization may use a reduced
heavy-atom representation during DPD relaxation, but the present implementation
keeps all atoms active so demo systems preserve direct all-atom coordinates.

Recommended MD handoff
----------------------
AA-DPD placement should be treated as an initialization step, not an equilibrated
production state. A typical handoff is:

1. Export the updated atom coordinates and periodic box to the target atomistic
   MD engine.
2. Build an all-atom force-field system with explicit hydrogens, periodic box
   vectors, and production-quality partial charges. For OpenFF validation of
   polyethylene, the NAGL/AshGC model ``openff-gnn-am1bcc-1.0.0.pt`` was used.
3. Run energy minimization and require finite energies.
4. Run short NVT dynamics with regular MD settings. For constrained hydrogens,
   ``2 fs`` and ``1 / ps`` Langevin friction are reasonable smoke-test settings.
5. Run NPT equilibration at the intended temperature and pressure until density,
   volume, potential energy, and kinetic energy are bounded and stationary.
6. Use the equilibrated NPT density for later NVT production if cleaner
   structural or dynamical statistics are needed.

"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .base import PlacementGenerator
from .random_walk import AngleConstrainedRandomWalk
from ..geometry.shapes import PointCloud
from ..interfaces._shared.topology import (
    build_saamr_role_topology_index,
    connector_reference_sort_key,
    resolve_to_atom_cached,
)
from ..mupr.primitives import Primitive
from ..roles import PrimitiveRole

LOGGER = logging.getLogger(__name__)

AMU_TO_G = 1.66053906660e-24
ANGSTROM3_TO_CM3 = 1.0e-24


@dataclass
class AllAtomDPDSettings:
    """Settings for :class:`AllAtomDPDBuilder`.

    Parameters
    ----------
    density_g_cm3
        Target mass density used to size the cubic simulation box unless
        ``box_lengths_a`` is supplied.
    box_lengths_a
        Optional explicit orthorhombic box lengths in Angstrom. Supplying these
        selects the finite-geometry call path: the caller owns chain selection
        and system construction, and AA-DPD relaxes the supplied hierarchy in
        the requested box.
    r_cut_a
        DPD pair cutoff in Angstrom.
    kT
        HOOMD thermostat temperature in the chosen reduced energy units.
    A_base
        Base DPD conservative repulsion prefactor.
    gamma_base
        Base DPD dissipative friction prefactor.
    dt
        HOOMD integration timestep.
    particle_spacing_a
        Minimum nearest-neighbor spacing required for convergence.
    initial_residue_spacing_a
        Target spacing between residue placements during frame-0 initialization.
    initial_angle_max_rad
        Maximum turn angle for the default random-walk residue placement.
    n_steps_per_interval
        Number of HOOMD steps between convergence checks.
    n_steps_max
        Maximum number of HOOMD steps.
    report_interval
        Interval for debug logging and optional trajectory writes.
    device
        HOOMD device selection: ``"auto"``, ``"CPU"``, or ``"GPU"``.
    force_field
        OpenFF force field identifier passed to ``ForceField``.
    bond_scale, angle_scale, dihedral_scale
        Multipliers applied to OpenFF bonded force constants. Defaults are
        intentionally stronger than production OpenFF values because AA-DPD uses
        them as dense-coordinate initialization restraints against large DPD
        packing forces.
    bond_energy_tolerance_a
        RMS-like bond displacement tolerance used to derive a harmonic bond
        energy threshold for AA-DPD convergence.
    angle_energy_tolerance_deg
        RMS-like angle displacement tolerance used to derive a harmonic angle
        energy threshold for AA-DPD convergence.
    require_bonded_energy_convergence
        Whether AA-DPD convergence requires bond and angle energies to fall below
        the derived thresholds in addition to nonbonded spacing.
    epsilon_reference_mode
        How to reduce OpenFF vdW epsilons into the DPD pair scaling reference.
    random_seed
        Optional deterministic seed for initialization and HOOMD.
    write_gsd
        Whether to write initial and trajectory GSD files.
    output_name
        Prefix for optional GSD output.
    resname_map
        Residue label to three-character residue name map for RDKit/OpenFF export.
    """

    density_g_cm3: float = 0.85
    box_lengths_a: Optional[tuple[float, float, float]] = None
    r_cut_a: float = 3.5
    kT: float = 1.0
    A_base: float = 5000.0
    gamma_base: float = 800.0
    dt: float = 0.001
    particle_spacing_a: float = 0.75
    initial_residue_spacing_a: float = 1.5
    initial_angle_max_rad: float = np.pi / 4.0
    n_steps_per_interval: int = 1000
    n_steps_max: int = 10000
    report_interval: int = 1000
    device: str = "auto"
    force_field: str = "openff-2.2.1.offxml"
    bond_scale: float = 30.0
    angle_scale: float = 30.0
    dihedral_scale: float = 30.0
    bond_energy_tolerance_a: float = 0.05
    angle_energy_tolerance_deg: float = 5.0
    require_bonded_energy_convergence: bool = True
    epsilon_reference_mode: str = "max"
    random_seed: Optional[int] = None
    write_gsd: bool = False
    output_name: Optional[str] = None
    resname_map: dict[str, str] = field(default_factory=dict)


@dataclass
class AllAtomDPDResult:
    """Summary of an all-atom DPD build."""

    atoms: list[Primitive]
    bonds: list[tuple[int, int]]
    angles: list[tuple[int, int, int]]
    dihedrals: list[tuple[int, int, int, int]]
    impropers: list[tuple[int, int, int, int]]
    particle_types: list[str]
    steps: int
    elapsed_s: float
    box_length_a: float
    box_lengths_a: tuple[float, float, float]
    converged: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AllAtomDPDBoxFillPlan:
    """Simple uniform chain-length plan for filling an explicit AA-DPD box.

    This is intentionally a planning utility, not a chemistry builder: callers
    still construct the MuPT hierarchy from their residue templates. A later
    extension can replace the uniform min/max sampling with a PDI-aware chain
    length distribution.
    """

    chain_lengths: list[int]
    target_mass_amu: float
    planned_mass_amu: float
    box_lengths_a: tuple[float, float, float]
    density_g_cm3: float


@dataclass
class _SegmentRecord:
    """Topology and atom-index mapping for one SEGMENT-role Primitive."""

    segment: Primitive
    residues: list[Primitive]
    atoms: list[Primitive]
    residue_atom_indices: list[list[int]]
    local_to_global: dict[int, int]
    bonds: list[tuple[int, int]]


@dataclass
class _ParameterTables:
    """HOOMD-ready bonded and vdW parameter tables."""

    bond_params: dict[str, dict[str, float]] = field(default_factory=dict)
    angle_params: dict[str, dict[str, float]] = field(default_factory=dict)
    dihedral_params: dict[str, dict[str, float]] = field(default_factory=dict)
    improper_params: dict[str, dict[str, float]] = field(default_factory=dict)
    bond_type_by_group: dict[tuple[int, int], str] = field(default_factory=dict)
    angle_type_by_group: dict[tuple[int, int, int], str] = field(default_factory=dict)
    dihedral_type_by_group: dict[tuple[int, int, int, int], list[str]] = field(default_factory=dict)
    improper_type_by_group: dict[tuple[int, int, int, int], list[str]] = field(default_factory=dict)
    atom_epsilons: dict[int, float] = field(default_factory=dict)
    atom_types_by_global: dict[int, str] = field(default_factory=dict)
    epsilon_by_type: dict[str, float] = field(default_factory=dict)


class AllAtomDPDParameterProvider(ABC):
    """Abstract source of all-atom DPD bonded and vdW parameter tables.

    ``AllAtomDPDBuilder`` consumes already-resolved HOOMD-style parameter
    tables through this boundary. OpenFF is the first provider implemented here,
    but callers can supply another provider that derives equivalent tables from
    a different force field, cached labels, or external parameterization step.
    """

    @abstractmethod
    def parameterize(
        self,
        root: Primitive,
        records: list[_SegmentRecord],
        settings: AllAtomDPDSettings,
    ) -> _ParameterTables:
        """Return HOOMD-ready parameters for the supplied segment records.

        Implementations should fill bonded parameter dictionaries and atom vdW
        epsilon/type mappings for every atom record they can parameterize.
        """


class OpenFFAllAtomDPDParameterProvider(AllAtomDPDParameterProvider):
    """Parameter provider backed by OpenFF ``ForceField.label_molecules``.

    OpenFF bonded terms are converted to numeric kcal/mol-style values and used
    as initialization restraints in HOOMD. This provider is intentionally one
    implementation of the ``AllAtomDPDParameterProvider`` interface rather than a
    requirement that every AA-DPD workflow use OpenFF internally.
    """

    def __init__(self, force_field: Optional[str] = None, resname_map: Optional[dict[str, str]] = None) -> None:
        """Create an OpenFF-backed parameter provider.

        Parameters
        ----------
        force_field
            Optional force-field identifier overriding builder settings.
        resname_map
            Optional residue-name map overriding builder settings.
        """

        self.force_field = force_field
        self.resname_map = resname_map

    def parameterize(
        self,
        root: Primitive,
        records: list[_SegmentRecord],
        settings: AllAtomDPDSettings,
    ) -> _ParameterTables:
        """Label RDKit segment molecules with OpenFF and collect parameters.

        The returned tables are builder-facing numeric parameters; downstream
        AA-DPD code does not depend on OpenFF objects after this method returns.
        """

        from openff.toolkit import ForceField, Molecule, Topology
        from openff.units import unit

        from ..interfaces.rdkit import primitive_to_rdkit_mols

        resname_map = settings.resname_map if self.resname_map is None else self.resname_map
        force_field = ForceField(self.force_field or settings.force_field)
        rdkit_mols = list(
            primitive_to_rdkit_mols(
                root,
                resname_map=resname_map,
                default_atom_position=np.zeros(3),
            )
        )
        molecules = [
            Molecule.from_rdkit(
                mol,
                allow_undefined_stereo=True,
                hydrogens_are_explicit=True,
            )
            for mol in rdkit_mols
        ]
        labels_by_mol = force_field.label_molecules(Topology.from_molecules(molecules))
        tables = _ParameterTables()

        for record, labels in zip(records, labels_by_mol):
            self._collect_vdw(labels, record, tables, unit)
            self._collect_bonds(labels, record, tables, unit, settings.bond_scale)
            self._collect_angles(labels, record, tables, unit, settings.angle_scale)
            self._collect_dihedrals(labels, record, tables, unit, settings.dihedral_scale)
            self._collect_impropers(labels, record, tables, unit, settings.dihedral_scale)

        return tables

    @staticmethod
    def _quantity_value(value: Any, target_unit: Any, default: float) -> float:
        """Return a float from an OpenFF quantity, tolerating missing labels."""

        if value is None:
            return default
        if hasattr(value, "m_as"):
            return float(value.m_as(target_unit))
        return float(value)

    def _collect_vdw(
        self,
        labels: dict[str, dict],
        record: _SegmentRecord,
        tables: _ParameterTables,
        unit: Any,
    ) -> None:
        """Collect per-atom vdW epsilon labels for DPD pair scaling."""

        for key, parameter in labels.get("vdW", {}).items():
            local_idx = self._atom_indices_from_openff_key(key)[0]
            global_idx = record.local_to_global[int(local_idx)]
            atom = record.atoms[int(local_idx)]
            atom_type = f"{atom.element.symbol}_{getattr(parameter, 'id', 'vdW')}"
            epsilon = self._quantity_value(
                getattr(parameter, "epsilon", None), unit.kilocalorie_per_mole, 1.0
            )
            tables.atom_types_by_global[global_idx] = atom_type
            tables.atom_epsilons[global_idx] = epsilon
            tables.epsilon_by_type[atom_type] = epsilon

    def _collect_bonds(
        self, labels: dict[str, dict], record: _SegmentRecord, tables: _ParameterTables, unit: Any, scale: float
    ) -> None:
        """Collect harmonic bond parameters from OpenFF labels."""

        for key, parameter in labels.get("Bonds", {}).items():
            local_pair = self._atom_indices_from_openff_key(key)
            i, j = (record.local_to_global[int(local_pair[0])], record.local_to_global[int(local_pair[1])])
            name = getattr(parameter, "id", None) or f"b{i}-{j}"
            tables.bond_type_by_group[tuple(sorted((i, j)))] = str(name)
            tables.bond_params[str(name)] = {
                "r0": self._quantity_value(getattr(parameter, "length", None), unit.angstrom, 1.5),
                "k": scale * self._quantity_value(
                    getattr(parameter, "k", None), unit.kilocalorie_per_mole / unit.angstrom**2, 100.0
                ),
            }

    def _collect_angles(
        self, labels: dict[str, dict], record: _SegmentRecord, tables: _ParameterTables, unit: Any, scale: float
    ) -> None:
        """Collect harmonic angle parameters from OpenFF labels."""

        for key, parameter in labels.get("Angles", {}).items():
            local_triplet = self._atom_indices_from_openff_key(key)
            name = getattr(parameter, "id", None) or "a-" + "-".join(map(str, local_triplet))
            group = tuple(record.local_to_global[int(idx)] for idx in local_triplet)
            tables.angle_type_by_group[group] = str(name)
            tables.angle_type_by_group[tuple(reversed(group))] = str(name)
            tables.angle_params[str(name)] = {
                "t0": self._quantity_value(getattr(parameter, "angle", None), unit.radian, np.pi / 2),
                "k": scale * self._quantity_value(
                    getattr(parameter, "k", None), unit.kilocalorie_per_mole / unit.radian**2, 20.0
                ),
            }

    def _collect_dihedrals(
        self, labels: dict[str, dict], record: _SegmentRecord, tables: _ParameterTables, unit: Any, scale: float
    ) -> None:
        """Collect periodic torsion parameters from OpenFF labels."""

        for key, parameter in labels.get("ProperTorsions", {}).items():
            local_quad = self._atom_indices_from_openff_key(key)
            group = tuple(record.local_to_global[int(idx)] for idx in local_quad)
            periodicity = getattr(parameter, "periodicity", [1])
            phase = getattr(parameter, "phase", [0.0])
            k = getattr(parameter, "k", [1.0])
            idivf = self._periodic_idivf(parameter, len(k))
            base_name = getattr(parameter, "id", None) or "d-" + "-".join(map(str, local_quad))
            for term_idx, k_term in enumerate(k):
                name = f"{base_name}_{term_idx}"
                k_value = (
                    scale
                    * self._quantity_value(k_term, unit.kilocalorie_per_mole, 1.0)
                    / float(idivf[term_idx])
                )
                tables.dihedral_type_by_group.setdefault(group, []).append(name)
                tables.dihedral_type_by_group.setdefault(tuple(reversed(group)), []).append(name)
                tables.dihedral_params[name] = {
                    "k": abs(k_value),
                    "d": 1 if k_value >= 0 else -1,
                    "n": int(periodicity[term_idx]),
                    "phi0": self._quantity_value(phase[term_idx], unit.radian, 0.0),
                }

    def _collect_impropers(
        self, labels: dict[str, dict], record: _SegmentRecord, tables: _ParameterTables, unit: Any, scale: float
    ) -> None:
        """Collect periodic improper torsion parameters from OpenFF labels."""

        for key, parameter in labels.get("ImproperTorsions", {}).items():
            local_quad = self._atom_indices_from_openff_key(key)
            group = tuple(record.local_to_global[int(idx)] for idx in local_quad)
            periodicity = getattr(parameter, "periodicity", [1])
            phase = getattr(parameter, "phase", [0.0])
            k = getattr(parameter, "k", [1.0])
            idivf = self._periodic_idivf(parameter, len(k))
            base_name = getattr(parameter, "id", None) or "i-" + "-".join(map(str, local_quad))
            for term_idx, k_term in enumerate(k):
                name = f"{base_name}_{term_idx}"
                k_value = (
                    scale
                    * self._quantity_value(k_term, unit.kilocalorie_per_mole, 1.0)
                    / float(idivf[term_idx])
                )
                tables.improper_type_by_group.setdefault(group, []).append(name)
                tables.improper_params[name] = {
                    "k": abs(k_value),
                    "d": 1 if k_value >= 0 else -1,
                    "n": int(periodicity[term_idx]),
                    "chi0": self._quantity_value(phase[term_idx], unit.radian, 0.0),
                }

    @staticmethod
    def _periodic_idivf(parameter: Any, n_terms: int) -> list[float]:
        """Return OpenFF periodic torsion idivf values, defaulting missing values."""

        idivf = getattr(parameter, "idivf", None)
        if idivf is None:
            return [1.0] * n_terms
        return list(idivf)

    @staticmethod
    def _atom_indices_from_openff_key(key: Any) -> tuple[int, ...]:
        """Return atom indices from OpenFF label keys across toolkit versions."""

        if hasattr(key, "atom_indices"):
            return tuple(int(idx) for idx in key.atom_indices)
        if hasattr(key, "this_atom_index"):
            return (int(key.this_atom_index),)
        return tuple(int(idx) for idx in key)


class AllAtomDPDBuilder:
    """All-atom DPD builder for SAAMR Primitive hierarchies."""

    def __init__(
        self,
        settings: Optional[AllAtomDPDSettings] = None,
        parameter_provider: Optional[AllAtomDPDParameterProvider] = None,
        placement_generator: Optional[PlacementGenerator] = None,
        resname_map: Optional[dict[str, str]] = None,
    ) -> None:
        """Create an all-atom DPD builder.

        Parameters
        ----------
        settings
            Builder settings. Defaults are tuned for dense all-atom coordinate
            initialization, not production molecular dynamics.
        parameter_provider
            Provider for OpenFF/HOOMD parameter tables.
        placement_generator
            Optional ``PlacementGenerator`` used for frame-0 residue placement.
            AA-DPD delegates this construction step to the repository placement
            abstraction to avoid a duplicate chain initializer; AA-DPD remains
            responsible for dense all-atom relaxation, not residue-chain
            construction. If omitted, AA-DPD creates a deterministic
            ``AngleConstrainedRandomWalk`` from the builder settings.
        resname_map
            Optional residue-name map overriding ``settings.resname_map``.
        """

        self.settings = settings or AllAtomDPDSettings()
        if resname_map is not None:
            self.settings.resname_map = dict(resname_map)
        self._validate_settings()
        self.parameter_provider = parameter_provider or OpenFFAllAtomDPDParameterProvider()
        self._uses_default_placement_generator = placement_generator is None
        self.placement_generator = placement_generator

    def _validate_settings(self) -> None:
        """Reject invalid settings before optional HOOMD/OpenFF work starts."""

        positive_fields = {
            "density_g_cm3": self.settings.density_g_cm3,
            "r_cut_a": self.settings.r_cut_a,
            "kT": self.settings.kT,
            "A_base": self.settings.A_base,
            "gamma_base": self.settings.gamma_base,
            "dt": self.settings.dt,
            "particle_spacing_a": self.settings.particle_spacing_a,
            "initial_residue_spacing_a": self.settings.initial_residue_spacing_a,
            "initial_angle_max_rad": self.settings.initial_angle_max_rad,
            "bond_scale": self.settings.bond_scale,
            "angle_scale": self.settings.angle_scale,
            "dihedral_scale": self.settings.dihedral_scale,
            "bond_energy_tolerance_a": self.settings.bond_energy_tolerance_a,
            "angle_energy_tolerance_deg": self.settings.angle_energy_tolerance_deg,
        }
        for name, value in positive_fields.items():
            if value <= 0.0:
                raise ValueError(f"AA-DPD {name} must be positive.")
        if self.settings.initial_angle_max_rad > np.pi:
            raise ValueError("AA-DPD initial_angle_max_rad must be <= pi radians.")
        if self.settings.box_lengths_a is not None:
            try:
                box_lengths = tuple(float(length) for length in self.settings.box_lengths_a)
            except TypeError as exc:
                raise ValueError("AA-DPD box_lengths_a must contain three positive lengths.") from exc
            if len(box_lengths) != 3 or any(length <= 0.0 for length in box_lengths):
                raise ValueError("AA-DPD box_lengths_a must contain three positive lengths.")
            minimum_length = 3.0 * self.settings.r_cut_a
            if min(box_lengths) < minimum_length:
                raise ValueError(
                    "AA-DPD explicit box_lengths_a values must each be at least "
                    f"3 * r_cut_a ({minimum_length:.3f} A) for HOOMD neighbor-list safety."
                )
            self.settings.box_lengths_a = box_lengths
        if self.settings.n_steps_per_interval < 1:
            raise ValueError("AA-DPD n_steps_per_interval must be >= 1.")
        if self.settings.n_steps_max < 0:
            raise ValueError("AA-DPD n_steps_max must be >= 0.")
        if self.settings.report_interval < 1:
            raise ValueError("AA-DPD report_interval must be >= 1.")
        device = str(self.settings.device).lower()
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError("AA-DPD device must be 'auto', 'CPU', or 'GPU'.")
        self.settings.device = {"auto": "auto", "cpu": "CPU", "gpu": "GPU"}[device]
        if self.settings.epsilon_reference_mode not in {"max", "mean"}:
            try:
                reference = float(self.settings.epsilon_reference_mode)
            except (TypeError, ValueError) as exc:
                raise ValueError("AA-DPD epsilon_reference_mode must be 'max', 'mean', or a positive number.") from exc
            if reference <= 0.0:
                raise ValueError("AA-DPD epsilon_reference_mode numeric value must be positive.")

    def _default_placement_generator(self, rng: np.random.Generator, box_length: float | np.ndarray) -> PlacementGenerator:
        """Return the default frame-0 residue placement generator."""

        box_lengths = self._as_box_lengths(box_length)
        initial_point = rng.uniform(-box_lengths / 2.0, box_lengths / 2.0, size=3)
        return AngleConstrainedRandomWalk(
            bond_length=self.settings.initial_residue_spacing_a,
            angle_max_rad=self.settings.initial_angle_max_rad,
            initial_point=initial_point,
            initial_direction=self._random_unit_vector(rng),
            rng=rng,
        )

    def build(self, root: Primitive) -> AllAtomDPDResult:
        """Mutate atom leaf coordinates in-place using an all-atom DPD run."""

        records = self._segment_records(root)
        atoms = [atom for record in records for atom in record.atoms]
        if not atoms:
            raise ValueError("AA-DPD build requires at least one SEGMENT with atom PARTICLE leaves.")

        import freud
        import gsd.hoomd
        import hoomd

        bonds = [
            tuple(sorted((record.local_to_global[i], record.local_to_global[j])))
            for record in records
            for i, j in record.bonds
        ]
        angles = self._angles_from_bonds(len(atoms), bonds)
        dihedrals = self._dihedrals_from_bonds(len(atoms), bonds)
        masses = np.array([self._atom_mass_amu(atom) for atom in atoms], dtype=float)
        box_lengths = self._box_lengths_a(float(masses.sum()))
        box_length = self._effective_box_length_a(box_lengths)
        root.metadata["unit_cell_parameters"] = [float(length) for length in box_lengths] + [90.0, 90.0, 90.0]

        parameters = self.parameter_provider.parameterize(root, records, self.settings)
        angles = [group for group in angles if group in parameters.angle_type_by_group]
        dihedrals = [group for group in dihedrals if group in parameters.dihedral_type_by_group]
        impropers = list(parameters.improper_type_by_group)
        frame = self._initial_frame(
            gsd.hoomd.Frame,
            records,
            atoms,
            bonds,
            angles,
            dihedrals,
            impropers,
            masses,
            parameters,
            box_lengths,
        )
        simulation = self._simulation(hoomd, frame, bonds, angles, dihedrals, impropers, parameters)
        steps, elapsed_s, converged, diagnostics = self._run_until_converged(
            simulation,
            freud,
            frame,
            parameters,
            box_lengths,
            bonds,
        )
        final_positions = self._unwrap_positions(
            simulation.state.get_snapshot().particles.position[:],
            bonds,
            box_lengths,
        )
        self._write_positions(root, atoms, final_positions)

        particle_types = list(frame.particles.types)
        result = AllAtomDPDResult(
            atoms=atoms,
            bonds=bonds,
            angles=angles,
            dihedrals=dihedrals,
            impropers=impropers,
            particle_types=particle_types,
            steps=steps,
            elapsed_s=elapsed_s,
            box_length_a=box_length,
            box_lengths_a=tuple(float(length) for length in box_lengths),
            converged=converged,
            diagnostics=diagnostics,
        )
        root.metadata["all_atom_dpd_summary"] = self._serializable_summary(result)
        return result

    def _segment_records(self, root: Primitive) -> list[_SegmentRecord]:
        """Return atom and bond records in role-aware traversal order."""

        index = build_saamr_role_topology_index(root)
        endpoint_cache: dict[tuple[int, object, object], Primitive] = {}
        records = []
        next_global = 0
        for segment in index.segments:
            atoms = []
            residue_atom_indices = []
            for residue in index.residues_by_segment[id(segment)]:
                residue_atoms = list(index.particles_by_residue[id(residue)])
                residue_atom_indices.append(list(range(len(atoms), len(atoms) + len(residue_atoms))))
                atoms.extend(residue_atoms)
            atom_id_to_local = {id(atom): idx for idx, atom in enumerate(atoms)}
            local_to_global = {idx: next_global + idx for idx in range(len(atoms))}
            bonds = []
            seen = set()
            for node in index.bond_nodes_by_segment[id(segment)]:
                for conn_ref_pair in node.internal_connections:
                    conn_ref1, conn_ref2 = sorted(conn_ref_pair, key=connector_reference_sort_key)
                    atom1 = resolve_to_atom_cached(node, conn_ref1, endpoint_cache)
                    atom2 = resolve_to_atom_cached(node, conn_ref2, endpoint_cache)
                    pair = tuple(sorted((atom_id_to_local[id(atom1)], atom_id_to_local[id(atom2)])))
                    if pair not in seen:
                        bonds.append(pair)
                        seen.add(pair)
            records.append(
                _SegmentRecord(
                    segment=segment,
                    residues=list(index.residues_by_segment[id(segment)]),
                    atoms=atoms,
                    residue_atom_indices=residue_atom_indices,
                    local_to_global=local_to_global,
                    bonds=sorted(bonds),
                )
            )
            next_global += len(atoms)
        return records

    @staticmethod
    def _atom_mass_amu(atom: Primitive) -> float:
        """Return an atom mass in amu from its periodictable element."""

        mass = getattr(atom.element, "mass", None)
        if mass is None:
            raise ValueError(f"Atom '{atom.label}' has no element mass for density-based box sizing.")
        return float(mass)

    def _box_length_a(self, total_mass_amu: float) -> float:
        """Return cubic box length in Angstrom from total mass and target density."""

        volume_cm3 = total_mass_amu * AMU_TO_G / self.settings.density_g_cm3
        density_length = float((volume_cm3 / ANGSTROM3_TO_CM3) ** (1.0 / 3.0))
        minimum_length = 3.0 * self.settings.r_cut_a
        if density_length < minimum_length:
            LOGGER.warning(
                "Density-derived AA-DPD box %.3f A is smaller than %.3f A; "
                "expanding small-system box for HOOMD neighbor-list safety.",
                density_length,
                minimum_length,
            )
        return max(density_length, minimum_length)

    def _box_lengths_a(self, total_mass_amu: float) -> np.ndarray:
        """Return orthorhombic box lengths in Angstrom for the active path."""

        if self.settings.box_lengths_a is not None:
            return np.asarray(self.settings.box_lengths_a, dtype=float)
        box_length = self._box_length_a(total_mass_amu)
        return np.array([box_length, box_length, box_length], dtype=float)

    @staticmethod
    def _effective_box_length_a(box_lengths: np.ndarray) -> float:
        """Return a volume-equivalent cubic length for legacy summary fields."""

        return float(np.prod(box_lengths) ** (1.0 / 3.0))

    @staticmethod
    def target_mass_for_box(density_g_cm3: float, box_lengths_a: tuple[float, float, float]) -> float:
        """Return target mass in amu for a density and orthorhombic AA-DPD box."""

        if density_g_cm3 <= 0.0:
            raise ValueError("AA-DPD target density_g_cm3 must be positive.")
        box_lengths = AllAtomDPDBuilder._as_box_lengths(box_lengths_a)
        if np.any(box_lengths <= 0.0):
            raise ValueError("AA-DPD box_lengths_a must contain three positive lengths.")
        volume_a3 = float(np.prod(box_lengths))
        return density_g_cm3 * volume_a3 * ANGSTROM3_TO_CM3 / AMU_TO_G

    @staticmethod
    def plan_uniform_chain_lengths_for_box(
        density_g_cm3: float,
        box_lengths_a: tuple[float, float, float],
        repeat_unit_mass_amu: float,
        chain_length_min: int,
        chain_length_max: int,
        random_seed: Optional[int] = None,
    ) -> AllAtomDPDBoxFillPlan:
        """Plan uniform min/max chain lengths to approximately fill a box.

        The returned lengths are repeat-unit counts. Callers can then construct
        chemically explicit chains using their own head/mid/tail templates. This
        can later be extended to sample from a PDI-driven molecular-weight
        distribution instead of a uniform integer range.
        """

        if repeat_unit_mass_amu <= 0.0:
            raise ValueError("AA-DPD repeat_unit_mass_amu must be positive.")
        if chain_length_min < 1 or chain_length_max < chain_length_min:
            raise ValueError("AA-DPD chain length bounds must satisfy 1 <= min <= max.")
        box_lengths = tuple(float(length) for length in AllAtomDPDBuilder._as_box_lengths(box_lengths_a))
        target_mass = AllAtomDPDBuilder.target_mass_for_box(density_g_cm3, box_lengths)
        rng = np.random.default_rng(random_seed)
        chain_lengths: list[int] = []
        planned_mass = 0.0
        while planned_mass < target_mass:
            length = int(rng.integers(chain_length_min, chain_length_max + 1))
            chain_lengths.append(length)
            planned_mass += length * repeat_unit_mass_amu
        return AllAtomDPDBoxFillPlan(
            chain_lengths=chain_lengths,
            target_mass_amu=float(target_mass),
            planned_mass_amu=float(planned_mass),
            box_lengths_a=box_lengths,
            density_g_cm3=float(density_g_cm3),
        )

    @staticmethod
    def _angles_from_bonds(n_atoms: int, bonds: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
        """Enumerate unique graph angles from bond pairs."""

        neighbors = [set() for _ in range(n_atoms)]
        for i, j in bonds:
            neighbors[i].add(j)
            neighbors[j].add(i)
        angles = []
        for center, nbrs in enumerate(neighbors):
            ordered = sorted(nbrs)
            for pos, left in enumerate(ordered):
                for right in ordered[pos + 1:]:
                    angles.append((left, center, right))
        return angles

    @staticmethod
    def _dihedrals_from_bonds(n_atoms: int, bonds: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
        """Enumerate unique graph dihedrals from bond pairs."""

        neighbors = [set() for _ in range(n_atoms)]
        for i, j in bonds:
            neighbors[i].add(j)
            neighbors[j].add(i)
        dihedrals = set()
        for j, k in bonds:
            for i in neighbors[j] - {k}:
                for l in neighbors[k] - {j}:
                    quad = (i, j, k, l)
                    dihedrals.add(min(quad, tuple(reversed(quad))))
        return sorted(dihedrals)

    def _initial_frame(
        self,
        frame_cls: type,
        records: list[_SegmentRecord],
        atoms: list[Primitive],
        bonds: list[tuple[int, int]],
        angles: list[tuple[int, int, int]],
        dihedrals: list[tuple[int, int, int, int]],
        impropers: list[tuple[int, int, int, int]],
        masses: np.ndarray,
        parameters: _ParameterTables,
        box_lengths: np.ndarray,
    ) -> Any:
        """Build a HOOMD frame with atom particles and bonded groups."""

        rng = np.random.default_rng(self.settings.random_seed)
        frame = frame_cls()
        missing_types = [idx for idx in range(len(atoms)) if idx not in parameters.atom_types_by_global]
        if missing_types:
            raise ValueError(f"AA-DPD parameterization did not assign particle types for atom indices {missing_types}.")
        particle_types = sorted(set(parameters.atom_types_by_global.values()))
        type_id = {name: idx for idx, name in enumerate(particle_types)}
        frame.particles.N = len(atoms)
        frame.particles.types = particle_types
        frame.particles.typeid = np.array(
            [type_id[parameters.atom_types_by_global[idx]] for idx in range(len(atoms))],
            dtype=np.uint32,
        )
        frame.particles.mass = masses
        frame.particles.position = self._initial_positions(records, box_lengths, rng)
        frame.configuration.box = [float(length) for length in box_lengths] + [0.0, 0.0, 0.0]
        self._set_bonded_frame_data(frame.bonds, bonds, parameters.bond_type_by_group, width=2)
        self._set_bonded_frame_data(frame.angles, angles, parameters.angle_type_by_group, width=3)
        self._set_bonded_frame_data(frame.dihedrals, dihedrals, parameters.dihedral_type_by_group, width=4)
        self._set_bonded_frame_data(frame.impropers, impropers, parameters.improper_type_by_group, width=4)
        return frame

    def _initial_positions(
        self,
        records: list[_SegmentRecord],
        box_lengths: float | np.ndarray | None = None,
        rng: Optional[np.random.Generator] = None,
        *,
        box_length: Optional[float] = None,
    ) -> np.ndarray:
        """Place residue templates with the shared PlacementGenerator abstraction.

        AA-DPD only needs pre-HOOMD frame-0 coordinates before its dense
        relaxation stage. Delegating residue-chain construction keeps this
        builder cohesive with the repository placement APIs and avoids a second,
        AA-DPD-specific placement abstraction.
        """

        if box_lengths is None:
            if box_length is None:
                raise TypeError("AA-DPD _initial_positions requires box_lengths or box_length.")
            box_lengths = box_length
        elif box_length is not None:
            raise TypeError("AA-DPD _initial_positions accepts only one of box_lengths or box_length.")
        if rng is None:
            raise TypeError("AA-DPD _initial_positions requires rng.")
        box_lengths = self._as_box_lengths(box_lengths)
        n_atoms = sum(len(record.atoms) for record in records)
        positions = np.zeros((n_atoms, 3), dtype=float)
        for record in records:
            missing_shapes = [atom.label for atom in record.atoms if atom.shape is None]
            if missing_shapes:
                raise ValueError(
                    "AA-DPD initialization requires atom coordinates; missing shapes for "
                    f"{missing_shapes}."
                )
            placement_segment, residue_handles = self._placement_segment(record)

            for residue_handle, residue_local_indices in zip(residue_handles, record.residue_atom_indices):
                residue_template = placement_segment.children_by_handle[residue_handle]
                residue_atoms = self._particle_leaves(residue_template)
                if len(residue_atoms) != len(residue_local_indices):
                    raise ValueError(
                        "AA-DPD residue template atom count changed while preparing PlacementGenerator input."
                    )
                residue_template.shape = PointCloud(
                    positions=np.array([atom.shape.centroid for atom in residue_atoms], dtype=float)
                )

            if self._uses_default_placement_generator and len(residue_handles) == 1:
                residue_handle = residue_handles[0]
                residue_template = placement_segment.children_by_handle[residue_handle]
                target_centroid = rng.uniform(-box_lengths / 2.0, box_lengths / 2.0, size=3)
                translation = target_centroid - np.asarray(residue_template.shape.centroid, dtype=float)
                residue_atoms = self._particle_leaves(residue_template)
                for local_idx, atom in zip(record.residue_atom_indices[0], residue_atoms):
                    global_idx = record.local_to_global[local_idx]
                    positions[global_idx] = self._wrap(np.asarray(atom.shape.centroid, dtype=float) + translation, box_lengths)
                continue

            if self._uses_default_placement_generator:
                placement_generator = self._default_placement_generator(rng, box_lengths)
            else:
                placement_generator = self.placement_generator
                if placement_generator is None:
                    raise RuntimeError("AA-DPD placement generator was unexpectedly unset.")
            placements_by_handle = {}
            duplicate_handles = []
            unknown_handles = []
            expected_handles = set(residue_handles)
            for residue_handle, placement in placement_generator.generate_placements(placement_segment):
                if residue_handle not in expected_handles:
                    unknown_handles.append(residue_handle)
                    continue
                if residue_handle in placements_by_handle:
                    duplicate_handles.append(residue_handle)
                    continue
                placements_by_handle[residue_handle] = placement

            missing_handles = [handle for handle in residue_handles if handle not in placements_by_handle]
            if missing_handles or duplicate_handles or unknown_handles:
                raise ValueError(
                    "AA-DPD PlacementGenerator output must yield exactly one placement "
                    "for each immediate RESIDUE child handle; "
                    f"missing={missing_handles}, duplicates={duplicate_handles}, "
                    f"unknown={unknown_handles}."
                )

            for residue_handle in residue_handles:
                placement_segment.children_by_handle[residue_handle].rigidly_transform(placements_by_handle[residue_handle])

            for residue_handle, residue_local_indices in zip(residue_handles, record.residue_atom_indices):
                residue_atoms = self._particle_leaves(placement_segment.children_by_handle[residue_handle])
                for local_idx, atom in zip(residue_local_indices, residue_atoms):
                    global_idx = record.local_to_global[local_idx]
                    # HOOMD periodic snapshots require wrapped particle positions;
                    # bonded molecules are unwrapped again after relaxation.
                    positions[global_idx] = self._wrap(np.asarray(atom.shape.centroid, dtype=float), box_lengths)
        return positions

    def _placement_segment(self, record: _SegmentRecord) -> tuple[Primitive, list[object]]:
        """Return a direct-residue segment adapted for ``PlacementGenerator``.

        PlacementGenerator intentionally knows nothing about SAAMR roles: it
        places the immediate children of the Primitive it receives. AA-DPD is the
        role-aware layer, so it adapts arbitrary transparent hierarchy levels
        between SEGMENT and RESIDUE into a temporary direct-child segment. The
        temporary segment is only frame-0 scaffolding; final coordinates are read
        from its residue atom templates and written back to the original atom
        leaves after DPD relaxation.
        """

        placement_segment = self._copy_untransformed_preserving_roles(record.segment)
        # Preserve the role-aware residue traversal order from _segment_records().
        # Primitive.expand() can reparent transparent-node children in registry
        # order that differs from DFS order when direct RESIDUE children and
        # transparent grouping nodes are mixed under the same SEGMENT.
        residue_templates = self._role_descendants(placement_segment, PrimitiveRole.RESIDUE)
        if len(residue_templates) != len(record.residues):
            raise ValueError(
                "AA-DPD could not mirror role-aware RESIDUE traversal in the "
                "temporary PlacementGenerator segment."
            )
        while True:
            transparent_handles = [
                handle
                for handle, child in placement_segment.children_by_handle.items()
                if child.role != PrimitiveRole.RESIDUE
            ]
            if not transparent_handles:
                break
            for handle in transparent_handles:
                child = placement_segment.children_by_handle[handle]
                if child.is_leaf or child.role in {PrimitiveRole.SEGMENT, PrimitiveRole.PARTICLE}:
                    raise ValueError(
                        "AA-DPD frame-0 PlacementGenerator adaptation expects only "
                        "transparent grouping nodes between SEGMENT and RESIDUE roles."
                    )
                placement_segment.expand(handle)

        handle_by_child_id = {id(child): handle for handle, child in placement_segment.children_by_handle.items()}
        residue_handles = [handle_by_child_id[id(residue)] for residue in residue_templates if id(residue) in handle_by_child_id]
        if len(residue_handles) != len(record.residues):
            raise ValueError(
                "AA-DPD could not adapt role-aware residues into a direct-child "
                "PlacementGenerator segment. Check that every RESIDUE role under "
                "the SEGMENT survived transparent-node expansion."
            )
        return placement_segment, residue_handles

    @staticmethod
    def _copy_untransformed_preserving_roles(node: Primitive) -> Primitive:
        """Copy a placement scaffold without dropping SAAMR roles.

        ``Primitive._copy_untransformed()`` intentionally focuses on geometry,
        connectors, children, and topology. The AA-DPD adapter also needs role
        labels on the temporary scaffold so it can distinguish transparent
        grouping nodes from RESIDUE templates before handing direct children to a
        role-agnostic PlacementGenerator.
        """

        clone = node._copy_untransformed()

        def copy_roles(source: Primitive, target: Primitive) -> None:
            target.role = source.role
            for source_child, target_child in zip(source.children, target.children):
                copy_roles(source_child, target_child)

        copy_roles(node, clone)
        return clone

    @staticmethod
    def _role_descendants(node: Primitive, role: PrimitiveRole) -> list[Primitive]:
        """Return descendants with ``role`` in deterministic child traversal order."""

        matches = []
        for child in node.children:
            if child.role == role:
                matches.append(child)
            matches.extend(AllAtomDPDBuilder._role_descendants(child, role))
        return matches

    @staticmethod
    def _particle_leaves(node: Primitive) -> list[Primitive]:
        """Return PARTICLE leaves below ``node`` in deterministic child order."""

        if node.is_leaf:
            return [node]
        atoms = []
        for child in node.children:
            atoms.extend(AllAtomDPDBuilder._particle_leaves(child))
        return atoms

    @staticmethod
    def _set_bonded_frame_data(
        container: Any,
        groups: list[tuple],
        type_by_group: dict[tuple, str],
        width: int,
    ) -> None:
        """Populate a HOOMD bonded container with group and type ids."""

        expanded_groups = []
        group_types = []
        for group in groups:
            group_type = type_by_group.get(tuple(group), "default")
            if isinstance(group_type, list):
                for type_name in group_type:
                    expanded_groups.append(group)
                    group_types.append(type_name)
            else:
                expanded_groups.append(group)
                group_types.append(group_type)
        unique_types = sorted(set(group_types)) or ["default"]
        container.N = len(expanded_groups)
        container.types = unique_types
        container.group = np.array(expanded_groups, dtype=np.uint32) if expanded_groups else np.zeros((0, width), dtype=np.uint32)
        container.typeid = np.array([unique_types.index(group_type) for group_type in group_types], dtype=np.uint32)

    def _simulation(
        self,
        hoomd: Any,
        frame: Any,
        bonds: list[tuple[int, int]],
        angles: list[tuple[int, int, int]],
        dihedrals: list[tuple[int, int, int, int]],
        impropers: list[tuple[int, int, int, int]],
        parameters: _ParameterTables,
    ) -> Any:
        """Create and configure a HOOMD simulation from the initial frame."""

        integrator = hoomd.md.Integrator(dt=self.settings.dt)
        integrator.methods.append(hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All()))
        if bonds:
            harmonic = hoomd.md.bond.Harmonic()
            for name in frame.bonds.types:
                harmonic.params[name] = self._bonded_params_for(
                    name,
                    parameters.bond_params,
                    {"r0": 1.5, "k": 100.0},
                    "bond",
                )
            integrator.forces.append(harmonic)
        if angles:
            harmonic_angle = hoomd.md.angle.Harmonic()
            for name in frame.angles.types:
                harmonic_angle.params[name] = self._bonded_params_for(
                    name,
                    parameters.angle_params,
                    {"t0": np.pi / 2, "k": 20.0},
                    "angle",
                )
            integrator.forces.append(harmonic_angle)
        if dihedrals:
            periodic = hoomd.md.dihedral.Periodic()
            for name in frame.dihedrals.types:
                periodic.params[name] = self._bonded_params_for(
                    name,
                    parameters.dihedral_params,
                    {"k": 1.0, "d": 1, "n": 1, "phi0": 0.0},
                    "dihedral",
                )
            integrator.forces.append(periodic)
        if impropers:
            periodic_improper = hoomd.md.improper.Periodic()
            for name in frame.impropers.types:
                periodic_improper.params[name] = self._bonded_params_for(
                    name,
                    parameters.improper_params,
                    {"k": 1.0, "d": 1, "n": 1, "chi0": 0.0},
                    "improper",
                )
            integrator.forces.append(periodic_improper)

        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        dpd = hoomd.md.pair.DPD(nlist, default_r_cut=self.settings.r_cut_a, kT=self.settings.kT)
        pair_params = self._dpd_pair_params(frame.particles.types, parameters.epsilon_by_type)
        for pair, param in pair_params.items():
            dpd.params[pair] = param
        integrator.forces.append(dpd)

        simulation = hoomd.Simulation(device=self._hoomd_device(hoomd), seed=self.settings.random_seed or 1)
        simulation.operations.integrator = integrator
        simulation.create_state_from_snapshot(frame)
        if self.settings.write_gsd and self.settings.output_name:
            import gsd.hoomd

            with gsd.hoomd.open(name=f"{self.settings.output_name}_init.gsd", mode="w") as handle:
                handle.append(frame)
            simulation.operations.writers.append(
                hoomd.write.GSD(
                    trigger=hoomd.trigger.Periodic(self.settings.report_interval),
                    filename=f"{self.settings.output_name}_traj.gsd",
                )
            )
        return simulation

    @staticmethod
    def _bonded_params_for(
        name: str,
        params_by_type: dict[str, dict[str, float]],
        emergency_default: dict[str, float],
        term_kind: str,
    ) -> dict[str, float]:
        """Return bonded parameters, loudly falling back to the stiffest known set."""

        if name in params_by_type:
            return params_by_type[name]
        if params_by_type:
            fallback_name, fallback_params = max(
                params_by_type.items(),
                key=lambda item: float(item[1].get("k", float("-inf"))),
            )
            assigned = dict(fallback_params)
            LOGGER.warning(
                "AA-DPD missing OpenFF %s parameters for type %r; using maximum-k "
                "parameter set %r with values %s.",
                term_kind,
                name,
                fallback_name,
                assigned,
            )
            return assigned
        assigned = dict(emergency_default)
        LOGGER.warning(
            "AA-DPD missing OpenFF %s parameters for type %r and no %s parameters "
            "were available; using emergency defaults %s.",
            term_kind,
            name,
            term_kind,
            assigned,
        )
        return assigned

    def _hoomd_device(self, hoomd: Any) -> Any:
        """Return the requested HOOMD device object."""

        if self.settings.device == "CPU":
            return hoomd.device.CPU()
        if self.settings.device == "GPU":
            return hoomd.device.GPU()
        return hoomd.device.auto_select()

    def _dpd_pair_params(self, particle_types: list[str], epsilon_by_type: dict[str, float]) -> dict[tuple[str, str], dict[str, float]]:
        """Return DPD pair parameters scaled by a simple epsilon heuristic."""

        reducer = max if self.settings.epsilon_reference_mode == "max" else np.mean
        if self.settings.epsilon_reference_mode not in {"max", "mean"}:
            reference = float(self.settings.epsilon_reference_mode)
        else:
            type_eps_values = list(epsilon_by_type.values())
            reference = float(reducer(type_eps_values)) if type_eps_values else 1.0
        params = {}
        for i, type_i in enumerate(particle_types):
            for type_j in particle_types[i:]:
                epsilon_i = epsilon_by_type.get(type_i, 1.0)
                epsilon_j = epsilon_by_type.get(type_j, 1.0)
                scale = np.sqrt(epsilon_i * epsilon_j) / reference if reference > 0 else 1.0
                params[(type_i, type_j)] = {
                    "A": self.settings.A_base * scale,
                    "gamma": self.settings.gamma_base * scale,
                }
        return params

    @classmethod
    def _energy_diagnostics(
        cls,
        simulation: Any,
        frame: Any,
        parameters: Optional[_ParameterTables] = None,
        settings: Optional[AllAtomDPDSettings] = None,
    ) -> dict[str, Any]:
        """Return final HOOMD force energies and per-term normalizations."""

        force_by_kind = cls._force_by_kind(simulation)
        counts = {
            "bond": int(getattr(frame.bonds, "N", 0)),
            "angle": int(getattr(frame.angles, "N", 0)),
            "dihedral": int(getattr(frame.dihedrals, "N", 0)),
            "improper": int(getattr(frame.impropers, "N", 0)),
        }
        diagnostics: dict[str, Any] = {"counts": counts}
        for kind in ("bond", "angle", "dihedral", "improper", "dpd"):
            energy = cls._force_energy(force_by_kind.get(kind))
            diagnostics[f"{kind}_energy"] = energy
            if kind in counts:
                count = counts[kind]
                diagnostics[f"{kind}_energy_per_term"] = energy / count if energy is not None and count else None
        if parameters is not None and settings is not None:
            diagnostics["bond_energy_threshold"] = cls._bond_energy_threshold(
                frame,
                parameters,
                settings.bond_energy_tolerance_a,
            )
            diagnostics["angle_energy_threshold"] = cls._angle_energy_threshold(
                frame,
                parameters,
                settings.angle_energy_tolerance_deg,
            )
            diagnostics["bond_energy_converged"] = cls._energy_below_threshold(
                diagnostics["bond_energy"],
                diagnostics["bond_energy_threshold"],
            )
            diagnostics["angle_energy_converged"] = cls._energy_below_threshold(
                diagnostics["angle_energy"],
                diagnostics["angle_energy_threshold"],
            )
            diagnostics["bonded_energy_converged"] = bool(
                diagnostics["bond_energy_converged"] and diagnostics["angle_energy_converged"]
            )
        return diagnostics

    @staticmethod
    def _energy_below_threshold(energy: Optional[float], threshold: Optional[float]) -> bool:
        """Return whether an energy is finite and at or below a threshold."""

        if threshold is None:
            return True
        if energy is None:
            return False
        return bool(np.isfinite(energy) and energy <= threshold)

    @classmethod
    def _bond_energy_threshold(
        cls,
        frame: Any,
        parameters: _ParameterTables,
        tolerance_a: float,
    ) -> Optional[float]:
        """Return total harmonic bond energy threshold for convergence."""

        if int(getattr(frame.bonds, "N", 0)) == 0:
            return None
        threshold = 0.0
        for type_name in cls._container_type_names(frame.bonds):
            k = float(parameters.bond_params[type_name]["k"])
            threshold += 0.5 * k * tolerance_a**2
        return float(threshold)

    @classmethod
    def _angle_energy_threshold(
        cls,
        frame: Any,
        parameters: _ParameterTables,
        tolerance_deg: float,
    ) -> Optional[float]:
        """Return total harmonic angle energy threshold for convergence."""

        if int(getattr(frame.angles, "N", 0)) == 0:
            return None
        tolerance_rad = float(np.deg2rad(tolerance_deg))
        threshold = 0.0
        for type_name in cls._container_type_names(frame.angles):
            k = float(parameters.angle_params[type_name]["k"])
            threshold += 0.5 * k * tolerance_rad**2
        return float(threshold)

    @staticmethod
    def _container_type_names(container: Any) -> list[str]:
        """Return one bonded type name per group/term in a HOOMD container."""

        return [container.types[int(typeid)] for typeid in container.typeid]

    @staticmethod
    def _force_by_kind(simulation: Any) -> dict[str, Any]:
        """Group configured HOOMD force objects by the AA-DPD role they play."""

        integrator = getattr(getattr(simulation, "operations", None), "integrator", None)
        forces = getattr(integrator, "forces", []) if integrator is not None else []
        force_by_kind = {}
        for force in forces:
            module = force.__class__.__module__
            class_name = force.__class__.__name__
            if module.endswith(".bond") or ".bond." in module:
                force_by_kind["bond"] = force
            elif module.endswith(".angle") or ".angle." in module:
                force_by_kind["angle"] = force
            elif module.endswith(".dihedral") or ".dihedral." in module:
                force_by_kind["dihedral"] = force
            elif module.endswith(".improper") or ".improper." in module:
                force_by_kind["improper"] = force
            elif class_name == "DPD" or module.endswith(".pair") or ".pair." in module:
                force_by_kind["dpd"] = force
        return force_by_kind

    @staticmethod
    def _force_energy(force: Any) -> Optional[float]:
        """Return a HOOMD force energy as a float, or ``None`` when absent."""

        if force is None:
            return None
        try:
            return float(force.energy)
        except (TypeError, ValueError):
            return None

    def _run_until_converged(
        self,
        simulation: Any,
        freud: Any,
        frame: Any,
        parameters: _ParameterTables,
        box_lengths: np.ndarray,
        excluded_pairs: list[tuple[int, int]],
    ) -> tuple[int, float, bool, dict[str, Any]]:
        """Run HOOMD intervals until spacing and bonded energies converge."""

        box_lengths = self._as_box_lengths(box_lengths)
        start = time.perf_counter()
        steps = 0
        simulation.run(1)
        excluded = {tuple(sorted(pair)) for pair in excluded_pairs}
        spacing_converged = self._spacing_ok(simulation.state.get_snapshot(), freud, box_lengths, excluded)
        diagnostics = self._energy_diagnostics(simulation, frame, parameters, self.settings)
        converged = self._convergence_ok(spacing_converged, diagnostics)
        while not converged and steps < self.settings.n_steps_max:
            simulation.run(self.settings.n_steps_per_interval)
            steps += self.settings.n_steps_per_interval
            spacing_converged = self._spacing_ok(simulation.state.get_snapshot(), freud, box_lengths, excluded)
            diagnostics = self._energy_diagnostics(simulation, frame, parameters, self.settings)
            converged = self._convergence_ok(spacing_converged, diagnostics)
            if steps % self.settings.report_interval == 0:
                LOGGER.debug(
                    "Integrated %s all-atom DPD steps; spacing=%s bonded=%s",
                    steps,
                    spacing_converged,
                    diagnostics.get("bonded_energy_converged"),
                )
        diagnostics["spacing_converged"] = bool(spacing_converged)
        diagnostics["converged"] = bool(converged)
        return steps, time.perf_counter() - start, converged, diagnostics

    def _convergence_ok(self, spacing_converged: bool, diagnostics: dict[str, Any]) -> bool:
        """Return whether AA-DPD stopping criteria are satisfied."""

        if not spacing_converged:
            return False
        if not self.settings.require_bonded_energy_convergence:
            return True
        return bool(diagnostics.get("bonded_energy_converged", False))

    def _run_until_spaced(
        self,
        simulation: Any,
        freud: Any,
        box_lengths: np.ndarray,
        excluded_pairs: list[tuple[int, int]],
    ) -> tuple[int, float, bool]:
        """Run HOOMD intervals until nearest-neighbor spacing converges.

        This legacy helper is retained for focused tests and external callers.
        ``build()`` uses ``_run_until_converged()``.
        """

        box_lengths = self._as_box_lengths(box_lengths)
        start = time.perf_counter()
        steps = 0
        simulation.run(1)
        excluded = {tuple(sorted(pair)) for pair in excluded_pairs}
        converged = self._spacing_ok(simulation.state.get_snapshot(), freud, box_lengths, excluded)
        while not converged and steps < self.settings.n_steps_max:
            simulation.run(self.settings.n_steps_per_interval)
            steps += self.settings.n_steps_per_interval
            if steps % self.settings.report_interval == 0:
                LOGGER.debug("Integrated %s all-atom DPD steps", steps)
            converged = self._spacing_ok(simulation.state.get_snapshot(), freud, box_lengths, excluded)
        return steps, time.perf_counter() - start, converged

    def _spacing_ok(self, snapshot: Any, freud: Any, box_lengths: np.ndarray, excluded: set[tuple[int, int]]) -> bool:
        """Return whether non-excluded pairs exceed the spacing threshold."""

        positions = snapshot.particles.position[:]
        box_lengths = self._as_box_lengths(box_lengths)
        box = freud.box.Box(float(box_lengths[0]), float(box_lengths[1]), float(box_lengths[2]))
        query = freud.locality.AABBQuery(box, positions).query(
            positions,
            {"r_min": 0.0, "r_max": self.settings.particle_spacing_a, "exclude_ii": True},
        )
        neighbors = query.toNeighborList()
        if len(neighbors) == 0:
            return True
        for i, j in zip(neighbors.query_point_indices, neighbors.point_indices):
            if tuple(sorted((int(i), int(j)))) not in excluded:
                return False
        return True

    @staticmethod
    def _unwrap_positions(positions: np.ndarray, bonds: list[tuple[int, int]], box_lengths: float | np.ndarray) -> np.ndarray:
        """Unwrap coordinates along the bond graph using minimum-image edges."""

        if len(positions) == 0:
            return positions
        box_lengths = AllAtomDPDBuilder._as_box_lengths(box_lengths)
        neighbors = [[] for _ in range(len(positions))]
        for i, j in bonds:
            neighbors[i].append(j)
            neighbors[j].append(i)
        unwrapped = positions.copy()
        seen = set()
        for start in range(len(positions)):
            if start in seen:
                continue
            seen.add(start)
            queue = deque([start])
            while queue:
                i = queue.popleft()
                for j in neighbors[i]:
                    if j in seen:
                        continue
                    delta = positions[j] - positions[i]
                    delta -= box_lengths * np.round(delta / box_lengths)
                    unwrapped[j] = unwrapped[i] + delta
                    seen.add(j)
                    queue.append(j)
        return unwrapped

    @staticmethod
    def _write_positions(root: Primitive, atoms: list[Primitive], positions: np.ndarray) -> None:
        """Write atom PointCloud positions and recompute composite PointClouds."""

        for atom, position in zip(atoms, positions):
            atom.shape = PointCloud(positions=np.asarray(position, dtype=float))

        def visit(node: Primitive) -> np.ndarray:
            if node.is_leaf:
                return node.shape.positions
            child_positions = [visit(child) for child in node.children]
            if child_positions:
                positions_here = np.vstack(child_positions)
                node.shape = PointCloud(positions=positions_here)
                return positions_here
            return np.zeros((0, 3), dtype=float)

        visit(root)

    @staticmethod
    def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
        """Draw a deterministic random unit vector from a NumPy generator."""

        vector = rng.normal(size=3)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else np.array([1.0, 0.0, 0.0])

    @staticmethod
    def _wrap(position: np.ndarray, box_lengths: float | np.ndarray) -> np.ndarray:
        """Wrap one position into a centered orthorhombic periodic box."""

        box_lengths = AllAtomDPDBuilder._as_box_lengths(box_lengths)
        return position - box_lengths * np.floor((position + box_lengths / 2.0) / box_lengths)

    @staticmethod
    def _as_box_lengths(box_lengths: float | np.ndarray | tuple[float, float, float]) -> np.ndarray:
        """Return a length-3 box vector from a scalar or orthorhombic lengths."""

        array = np.asarray(box_lengths, dtype=float)
        if array.ndim == 0:
            return np.repeat(float(array), 3)
        if array.shape != (3,):
            raise ValueError("AA-DPD box lengths must be a scalar or a length-3 vector.")
        return array

    @staticmethod
    def _serializable_summary(result: AllAtomDPDResult) -> dict[str, Any]:
        """Return JSON-like metadata summary values from a result."""

        return {
            "n_atoms": len(result.atoms),
            "n_bonds": len(result.bonds),
            "n_angles": len(result.angles),
            "n_dihedrals": len(result.dihedrals),
            "n_impropers": len(result.impropers),
            "particle_types": list(result.particle_types),
            "steps": int(result.steps),
            "elapsed_s": float(result.elapsed_s),
            "box_length_a": float(result.box_length_a),
            "box_lengths_a": [float(length) for length in result.box_lengths_a],
            "converged": bool(result.converged),
            "diagnostics": result.diagnostics,
        }
