"""All-atom DPD coordinate builder for SAAMR-compliant Primitive hierarchies.

The builder uses OpenFF labels to construct bonded restraints and heuristic DPD
repulsions for dense coordinate initialization. The HOOMD simulation is meant to 
produce finite all-atom melt coordinates suitable for downstream minimization in 
an MD engine.

Recommended MD handoff
----------------------
AA-DPD placement should be treated as an initialization step, not an equilibrated
production state. A typical handoff is:

1. Export the updated atom coordinates and periodic box to the target atomistic
   MD engine.
2. Build an all-atom force-field system with explicit hydrogens, periodic box
   vectors, and production-quality partial charges. For OpenFF validation of
   polyethylene, the NAGL/AshGC model ``openff-gnn-am1bcc-1.0.0.pt`` was used/
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
from typing import Any, Callable, Optional

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
        Target mass density used to size the cubic simulation box.
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
    initial_bond_length_a
        Target spacing between residue placements during frame-0 initialization.
    initial_angle_max_rad
        Maximum turn angle for the default random-walk residue placement.
    n_steps_per_interval
        Number of HOOMD steps between convergence checks.
    n_steps_max
        Maximum number of HOOMD steps.
    report_interval
        Interval for debug logging and optional trajectory writes.
    force_field
        OpenFF force field identifier passed to ``ForceField``.
    bond_scale, angle_scale, dihedral_scale
        Multipliers applied to OpenFF bonded force constants.
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
    r_cut_a: float = 3.5
    kT: float = 1.0
    A_base: float = 5000.0
    gamma_base: float = 800.0
    dt: float = 0.001
    particle_spacing_a: float = 0.75
    initial_bond_length_a: float = 5.0
    initial_angle_max_rad: float = np.pi / 4.0
    n_steps_per_interval: int = 1000
    n_steps_max: int = 10000
    report_interval: int = 1000
    force_field: str = "openff-2.2.1.offxml"
    bond_scale: float = 1.0
    angle_scale: float = 1.0
    dihedral_scale: float = 1.0
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
    converged: bool


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
    """Abstract source of all-atom DPD bonded and vdW parameter tables."""

    @abstractmethod
    def parameterize(
        self,
        root: Primitive,
        records: list[_SegmentRecord],
        settings: AllAtomDPDSettings,
    ) -> _ParameterTables:
        """Return HOOMD-ready parameters for the supplied segment records."""


class OpenFFAllAtomDPDParameterProvider(AllAtomDPDParameterProvider):
    """Parameter provider backed by OpenFF ``ForceField.label_molecules``.

    OpenFF bonded terms are converted to numeric kcal/mol-style values and used
    as initialization restraints in HOOMD. 
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
        """Label RDKit segment molecules with OpenFF and collect parameters."""

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
            idivf = getattr(parameter, "idivf", [1.0] * len(k))
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
            idivf = getattr(parameter, "idivf", [1.0] * len(k))
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
                    "phi0": self._quantity_value(phase[term_idx], unit.radian, 0.0),
                }

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
        placement_generator_factory: Callable[[np.random.Generator, float], PlacementGenerator] | None = None,
        resname_map: Optional[dict[str, str]] = None,
    ) -> None:
        """Create an all-atom DPD builder.

        Parameters
        ----------
        settings
            Builder settings. Defaults mirror the Issue #77 notebook values.
        parameter_provider
            Provider for OpenFF/HOOMD parameter tables.
        placement_generator_factory
            Factory returning a ``PlacementGenerator`` for frame-0 residue
            placement from the AA-DPD RNG and box length. AA-DPD delegates this
            construction step to the repository placement abstraction to avoid a
            duplicate chain initializer; AA-DPD remains responsible for dense
            all-atom relaxation, not residue-chain construction.
        resname_map
            Optional residue-name map overriding ``settings.resname_map``.
        """

        self.settings = settings or AllAtomDPDSettings()
        if resname_map is not None:
            self.settings.resname_map = dict(resname_map)
        self._validate_settings()
        self.parameter_provider = parameter_provider or OpenFFAllAtomDPDParameterProvider()
        self.placement_generator_factory = placement_generator_factory or self._default_placement_generator

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
            "initial_bond_length_a": self.settings.initial_bond_length_a,
            "initial_angle_max_rad": self.settings.initial_angle_max_rad,
            "bond_scale": self.settings.bond_scale,
            "angle_scale": self.settings.angle_scale,
            "dihedral_scale": self.settings.dihedral_scale,
        }
        for name, value in positive_fields.items():
            if value <= 0.0:
                raise ValueError(f"AA-DPD {name} must be positive.")
        if self.settings.initial_angle_max_rad > np.pi:
            raise ValueError("AA-DPD initial_angle_max_rad must be <= pi radians.")
        if self.settings.n_steps_per_interval < 1:
            raise ValueError("AA-DPD n_steps_per_interval must be >= 1.")
        if self.settings.n_steps_max < 0:
            raise ValueError("AA-DPD n_steps_max must be >= 0.")
        if self.settings.report_interval < 1:
            raise ValueError("AA-DPD report_interval must be >= 1.")
        if self.settings.epsilon_reference_mode not in {"max", "mean"}:
            try:
                reference = float(self.settings.epsilon_reference_mode)
            except (TypeError, ValueError) as exc:
                raise ValueError("AA-DPD epsilon_reference_mode must be 'max', 'mean', or a positive number.") from exc
            if reference <= 0.0:
                raise ValueError("AA-DPD epsilon_reference_mode numeric value must be positive.")

    def _default_placement_generator(self, rng: np.random.Generator, box_length: float) -> PlacementGenerator:
        """Return the default frame-0 residue placement generator."""

        initial_point = rng.uniform(-box_length / 2.0, box_length / 2.0, size=3)
        return AngleConstrainedRandomWalk(
            bond_length=self.settings.initial_bond_length_a,
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
        box_length = self._box_length_a(float(masses.sum()))
        root.metadata["unit_cell_parameters"] = [box_length, box_length, box_length, 90.0, 90.0, 90.0]

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
            box_length,
        )
        simulation = self._simulation(hoomd, frame, bonds, angles, dihedrals, impropers, parameters)
        steps, elapsed_s, converged = self._run_until_spaced(simulation, freud, box_length, bonds)
        final_positions = self._unwrap_positions(
            simulation.state.get_snapshot().particles.position[:],
            bonds,
            box_length,
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
            converged=converged,
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
        box_length: float,
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
        frame.particles.position = self._initial_positions(records, box_length, rng)
        frame.configuration.box = [box_length, box_length, box_length, 0.0, 0.0, 0.0]
        self._set_bonded_frame_data(frame.bonds, bonds, parameters.bond_type_by_group, width=2)
        self._set_bonded_frame_data(frame.angles, angles, parameters.angle_type_by_group, width=3)
        self._set_bonded_frame_data(frame.dihedrals, dihedrals, parameters.dihedral_type_by_group, width=4)
        self._set_bonded_frame_data(frame.impropers, impropers, parameters.improper_type_by_group, width=4)
        return frame

    def _initial_positions(
        self,
        records: list[_SegmentRecord],
        box_length: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Place residue templates with the shared PlacementGenerator abstraction.

        AA-DPD only needs pre-HOOMD frame-0 coordinates before its dense
        relaxation stage. Delegating residue-chain construction keeps this
        builder cohesive with the repository placement APIs and avoids a second,
        AA-DPD-specific placement abstraction.
        """

        n_atoms = sum(len(record.atoms) for record in records)
        positions = np.zeros((n_atoms, 3), dtype=float)
        for record in records:
            missing_shapes = [atom.label for atom in record.atoms if atom.shape is None]
            if missing_shapes:
                raise ValueError(
                    "AA-DPD initialization requires atom coordinates; missing shapes for "
                    f"{missing_shapes}."
                )
            segment_template = record.segment._copy_untransformed()
            segment_child_items = list(record.segment.children_by_handle.items())
            invalid_children = [child.label for _handle, child in segment_child_items if child.role != PrimitiveRole.RESIDUE]
            residue_handles = [
                handle
                for handle, residue in segment_child_items
                if any(residue is expected_residue for expected_residue in record.residues)
            ]
            if invalid_children or len(residue_handles) != len(record.residues) or len(residue_handles) != len(segment_template.children):
                raise ValueError(
                    "AA-DPD frame-0 PlacementGenerator initialization requires every "
                    "immediate child of each SEGMENT to be a RESIDUE-role Primitive. "
                    "PlacementGenerator places direct children only; move transparent "
                    "grouping nodes above SEGMENT or provide a custom "
                    "placement_generator_factory for nested layouts. "
                    f"Invalid immediate SEGMENT children: {invalid_children}."
                )

            for residue_handle, residue_local_indices in zip(residue_handles, record.residue_atom_indices):
                residue_template = segment_template.children_by_handle[residue_handle]
                residue_atoms = self._particle_leaves(residue_template)
                if len(residue_atoms) != len(residue_local_indices):
                    raise ValueError(
                        "AA-DPD residue template atom count changed while preparing PlacementGenerator input."
                    )
                residue_template.shape = PointCloud(
                    positions=np.array([atom.shape.centroid for atom in residue_atoms], dtype=float)
                )

            placement_generator = self.placement_generator_factory(rng, box_length)
            placements_by_handle = {}
            duplicate_handles = []
            unknown_handles = []
            expected_handles = set(residue_handles)
            for residue_handle, placement in placement_generator.generate_placements(segment_template):
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
                segment_template.children_by_handle[residue_handle].rigidly_transform(placements_by_handle[residue_handle])

            for residue_handle, residue_local_indices in zip(residue_handles, record.residue_atom_indices):
                residue_atoms = self._particle_leaves(segment_template.children_by_handle[residue_handle])
                for local_idx, atom in zip(residue_local_indices, residue_atoms):
                    global_idx = record.local_to_global[local_idx]
                    # HOOMD periodic snapshots require wrapped particle positions;
                    # bonded molecules are unwrapped again after relaxation.
                    positions[global_idx] = self._wrap(np.asarray(atom.shape.centroid, dtype=float), box_length)
        return positions

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
                harmonic.params[name] = parameters.bond_params.get(name, {"r0": 1.5, "k": 100.0})
            integrator.forces.append(harmonic)
        if angles:
            harmonic_angle = hoomd.md.angle.Harmonic()
            for name in frame.angles.types:
                harmonic_angle.params[name] = parameters.angle_params.get(name, {"t0": np.pi / 2, "k": 20.0})
            integrator.forces.append(harmonic_angle)
        if dihedrals:
            periodic = hoomd.md.dihedral.Periodic()
            for name in frame.dihedrals.types:
                periodic.params[name] = parameters.dihedral_params.get(name, {"k": 1.0, "d": 1, "n": 1, "phi0": 0.0})
            integrator.forces.append(periodic)
        if impropers:
            periodic_improper = hoomd.md.improper.Periodic()
            for name in frame.impropers.types:
                periodic_improper.params[name] = parameters.improper_params.get(name, {"k": 1.0, "d": 1, "n": 1, "phi0": 0.0})
            integrator.forces.append(periodic_improper)

        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        dpd = hoomd.md.pair.DPD(nlist, default_r_cut=self.settings.r_cut_a, kT=self.settings.kT)
        pair_params = self._dpd_pair_params(frame.particles.types, parameters.epsilon_by_type)
        for pair, param in pair_params.items():
            dpd.params[pair] = param
        integrator.forces.append(dpd)

        simulation = hoomd.Simulation(device=hoomd.device.auto_select(), seed=self.settings.random_seed or 1)
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

    def _run_until_spaced(
        self,
        simulation: Any,
        freud: Any,
        box_length: float,
        excluded_pairs: list[tuple[int, int]],
    ) -> tuple[int, float, bool]:
        """Run HOOMD intervals until nearest-neighbor spacing converges."""

        start = time.perf_counter()
        steps = 0
        simulation.run(1)
        excluded = {tuple(sorted(pair)) for pair in excluded_pairs}
        converged = self._spacing_ok(simulation.state.get_snapshot(), freud, box_length, excluded)
        while not converged and steps < self.settings.n_steps_max:
            simulation.run(self.settings.n_steps_per_interval)
            steps += self.settings.n_steps_per_interval
            if steps % self.settings.report_interval == 0:
                LOGGER.debug("Integrated %s all-atom DPD steps", steps)
            converged = self._spacing_ok(simulation.state.get_snapshot(), freud, box_length, excluded)
        return steps, time.perf_counter() - start, converged

    def _spacing_ok(self, snapshot: Any, freud: Any, box_length: float, excluded: set[tuple[int, int]]) -> bool:
        """Return whether non-excluded pairs exceed the spacing threshold."""

        positions = snapshot.particles.position[:]
        box = freud.box.Box.cube(box_length)
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
    def _unwrap_positions(positions: np.ndarray, bonds: list[tuple[int, int]], box_length: float) -> np.ndarray:
        """Unwrap coordinates along the bond graph using minimum-image edges."""

        if len(positions) == 0:
            return positions
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
                    delta -= box_length * np.round(delta / box_length)
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
    def _wrap(position: np.ndarray, box_length: float) -> np.ndarray:
        """Wrap one position into a centered cubic periodic box."""

        return position - box_length * np.floor((position + box_length / 2.0) / box_length)

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
            "converged": bool(result.converged),
        }
