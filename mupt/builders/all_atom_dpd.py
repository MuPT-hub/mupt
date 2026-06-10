"""All-atom DPD coordinate builder for SAAMR Primitive hierarchies.

The builder uses OpenFF labels to construct bonded restraints and heuristic DPD
repulsions for dense coordinate initialization. The HOOMD simulation is not a
calibrated physical DPD model; its contract is to produce finite all-atom melt
coordinates suitable for downstream minimization in an MD engine.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry.shapes import PointCloud
from ..interfaces._shared.topology import (
    build_saamr_role_topology_index,
    connector_reference_sort_key,
    resolve_to_atom_cached,
)
from ..mupr.primitives import Primitive

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
    initial_chain_step_a
        Distance between segment template centers during initialization.
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
    initial_chain_step_a: float = 5.0
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
    particle_types: list[str]
    steps: int
    elapsed_s: float
    box_length_a: float
    converged: bool


@dataclass
class _SegmentRecord:
    """Topology and atom-index mapping for one SEGMENT-role Primitive."""

    segment: Primitive
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
    bond_type_by_group: dict[tuple[int, int], str] = field(default_factory=dict)
    angle_type_by_group: dict[tuple[int, int, int], str] = field(default_factory=dict)
    dihedral_type_by_group: dict[tuple[int, int, int, int], str] = field(default_factory=dict)
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
    as initialization restraints in HOOMD. They should not be interpreted as a
    validated HOOMD/OpenFF unit conversion for production dynamics.
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
            name = getattr(parameter, "id", None) or "d-" + "-".join(map(str, local_quad))
            group = tuple(record.local_to_global[int(idx)] for idx in local_quad)
            periodicity = getattr(parameter, "periodicity", [1])
            phase = getattr(parameter, "phase", [0.0])
            k = getattr(parameter, "k", [1.0])
            tables.dihedral_type_by_group[group] = str(name)
            tables.dihedral_type_by_group[tuple(reversed(group))] = str(name)
            tables.dihedral_params[str(name)] = {
                "k": scale * self._quantity_value(k[0], unit.kilocalorie_per_mole, 1.0),
                "d": 1 if self._quantity_value(phase[0], unit.radian, 0.0) < np.pi / 2 else -1,
                "n": int(periodicity[0]),
                "phi0": self._quantity_value(phase[0], unit.radian, 0.0),
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
    """Global all-atom DPD builder for SAAMR Primitive hierarchies."""

    def __init__(
        self,
        settings: Optional[AllAtomDPDSettings] = None,
        parameter_provider: Optional[AllAtomDPDParameterProvider] = None,
        resname_map: Optional[dict[str, str]] = None,
    ) -> None:
        """Create an all-atom DPD builder.

        Parameters
        ----------
        settings
            Builder settings. Defaults mirror the Issue #77 notebook values.
        parameter_provider
            Provider for OpenFF/HOOMD parameter tables.
        resname_map
            Optional residue-name map overriding ``settings.resname_map``.
        """

        self.settings = settings or AllAtomDPDSettings()
        if resname_map is not None:
            self.settings.resname_map = dict(resname_map)
        if self.settings.density_g_cm3 <= 0.0:
            raise ValueError("AA-DPD density_g_cm3 must be positive.")
        self.parameter_provider = parameter_provider or OpenFFAllAtomDPDParameterProvider()

    def build(self, root: Primitive) -> AllAtomDPDResult:
        """Mutate atom leaf coordinates in-place using a global all-atom DPD run."""

        import freud
        import gsd.hoomd
        import hoomd

        records = self._segment_records(root)
        atoms = [atom for record in records for atom in record.atoms]
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
        frame = self._initial_frame(
            gsd.hoomd.Frame,
            records,
            atoms,
            bonds,
            angles,
            dihedrals,
            masses,
            parameters,
            box_length,
        )
        simulation = self._simulation(hoomd, frame, bonds, angles, dihedrals, parameters)
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
        masses: np.ndarray,
        parameters: _ParameterTables,
        box_length: float,
    ) -> Any:
        """Build a GSD frame with atom particles and bonded groups."""

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
        return frame

    def _initial_positions(
        self,
        records: list[_SegmentRecord],
        box_length: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Place intact residue templates as simple random-orientation chains."""

        n_atoms = sum(len(record.atoms) for record in records)
        positions = np.zeros((n_atoms, 3), dtype=float)
        for record in records:
            missing_shapes = [atom.label for atom in record.atoms if atom.shape is None]
            if missing_shapes:
                raise ValueError(
                    "AA-DPD initialization requires atom coordinates; missing shapes for "
                    f"{missing_shapes}."
                )
            local = np.array([atom.shape.centroid for atom in record.atoms], dtype=float)
            center = rng.uniform(-box_length / 2.0, box_length / 2.0, size=3)
            rotation = Rotation.random(random_state=rng)
            center_offset = 0.5 * (len(record.residue_atom_indices) - 1) * self.settings.initial_chain_step_a
            for residue_idx, residue_local_indices in enumerate(record.residue_atom_indices):
                residue_positions = local[residue_local_indices]
                residue_center = residue_positions.mean(axis=0)
                chain_offset = np.array(
                    [residue_idx * self.settings.initial_chain_step_a - center_offset, 0.0, 0.0],
                    dtype=float,
                )
                for local_idx in residue_local_indices:
                    global_idx = record.local_to_global[local_idx]
                    local_offset = local[local_idx] - residue_center
                    positions[global_idx] = self._wrap(center + rotation.apply(chain_offset + local_offset), box_length)
        return positions

    @staticmethod
    def _set_bonded_frame_data(
        container: Any,
        groups: list[tuple],
        type_by_group: dict[tuple, str],
        width: int,
    ) -> None:
        """Populate a GSD bonded container with group and type ids."""

        group_types = [type_by_group.get(tuple(group), "default") for group in groups]
        unique_types = sorted(set(group_types)) or ["default"]
        container.N = len(groups)
        container.types = unique_types
        container.group = np.array(groups, dtype=np.uint32) if groups else np.zeros((0, width), dtype=np.uint32)
        container.typeid = np.array([unique_types.index(group_type) for group_type in group_types], dtype=np.uint32)

    def _simulation(
        self,
        hoomd: Any,
        frame: Any,
        bonds: list[tuple[int, int]],
        angles: list[tuple[int, int, int]],
        dihedrals: list[tuple[int, int, int, int]],
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
        """Write atom PointCloud shapes and recompute composite PointClouds."""

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
            "particle_types": list(result.particle_types),
            "steps": int(result.steps),
            "elapsed_s": float(result.elapsed_s),
            "box_length_a": float(result.box_length_a),
            "converged": bool(result.converged),
        }
