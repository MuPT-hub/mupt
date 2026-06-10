#!/usr/bin/env python
"""Manual dense polyethylene AA-DPD/OpenMM smoke-test harness."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings
from mupt.tests.conftest import build_SAAMR_polymer_system


AMU_TO_G = 1.66053906660e-24
ANGSTROM3_TO_CM3 = 1.0e-24
PE_SMILES = {
    "head": "[H:1]-[CH2:2]-*",
    "ethane": "*-[CH2:1][CH2:2]-*",
    "tail": "*-[CH2:1]-[H:2]",
}
PE_RESNAME_MAP = {"head": "HEA", "ethane": "EAN", "tail": "TYL"}


@dataclass(frozen=True)
class OpenMMDeps:
    """Optional OpenMM validation imports."""

    Interchange: Any
    ForceField: Any
    Molecule: Any
    Topology: Any
    off_unit: Any
    omm_unit: Any
    LangevinIntegrator: Any
    Vec3: Any
    Simulation: Any
    primitive_to_rdkit_mols: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic dense polyethylene melt, run the all-atom "
            "DPD builder, and optionally smoke-test OpenMM minimization."
        )
    )
    parser.add_argument("--n-chains", type=int, default=10, help="Number of PE chains to build.")
    parser.add_argument("--chain-len", type=int, default=15, help="Repeat units per PE chain.")
    parser.add_argument("--density-g-cm3", type=float, default=0.85, help="Target melt density.")
    parser.add_argument("--dpd-max-steps", type=int, default=2000, help="Maximum DPD integration steps.")
    parser.add_argument(
        "--dpd-steps-per-interval",
        type=int,
        default=250,
        help="DPD steps between convergence checks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic build and DPD seed.")
    parser.add_argument("--skip-openmm", action="store_true", help="Skip OpenMM minimization smoke test.")
    parser.add_argument(
        "--openmm-max-iterations",
        type=int,
        default=100,
        help="Maximum OpenMM minimization iterations.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_chains < 1:
        raise ValueError("--n-chains must be >= 1")
    if args.chain_len < 2:
        raise ValueError("--chain-len must be >= 2 because PE needs head and tail units")
    if args.density_g_cm3 <= 0.0:
        raise ValueError("--density-g-cm3 must be > 0")
    if args.dpd_max_steps < 0:
        raise ValueError("--dpd-max-steps must be >= 0")
    if args.dpd_steps_per_interval < 1:
        raise ValueError("--dpd-steps-per-interval must be >= 1")
    if args.openmm_max_iterations < 0:
        raise ValueError("--openmm-max-iterations must be >= 0")


def build_pe_melt(args: argparse.Namespace) -> Any:
    return build_SAAMR_polymer_system(
        PE_SMILES,
        mid_distrib={"ethane": 1.0},
        n_chains=args.n_chains,
        chain_len_min=args.chain_len,
        chain_len_max=args.chain_len,
        random_seed=args.seed,
    )


def run_dpd(root: Any, args: argparse.Namespace) -> Any:
    settings = AllAtomDPDSettings(
        density_g_cm3=args.density_g_cm3,
        n_steps_max=args.dpd_max_steps,
        n_steps_per_interval=args.dpd_steps_per_interval,
        report_interval=args.dpd_steps_per_interval,
        random_seed=args.seed,
        write_gsd=False,
        resname_map=dict(PE_RESNAME_MAP),
    )
    try:
        return AllAtomDPDBuilder(settings=settings, resname_map=PE_RESNAME_MAP).build(root)
    except ModuleNotFoundError as exc:
        missing = exc.name or "an optional AA-DPD dependency"
        raise RuntimeError(
            "AA-DPD validation dependencies are missing. Install hoomd, freud, "
            "gsd, openff-toolkit, and their runtime dependencies in this environment. "
            f"Missing import: {missing}"
        ) from exc


def atom_positions(atoms: list[Any]) -> np.ndarray:
    positions = []
    for atom in atoms:
        if atom.shape is None:
            positions.append(np.full(3, np.nan, dtype=float))
        else:
            positions.append(np.asarray(atom.shape.centroid, dtype=float))
    return np.asarray(positions, dtype=float)


def total_mass_amu(atoms: list[Any]) -> float:
    return float(sum(float(atom.element.mass) for atom in atoms))


def density_g_cm3(total_mass: float, box_length_a: float) -> float:
    volume_cm3 = box_length_a**3 * ANGSTROM3_TO_CM3
    return total_mass * AMU_TO_G / volume_cm3


def min_distinct_distance_a(positions: np.ndarray) -> float:
    if len(positions) < 2 or not np.all(np.isfinite(positions)):
        return math.nan
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    distances[np.tril_indices(len(positions))] = np.inf
    return float(np.min(distances))


def print_dpd_diagnostics(result: Any) -> None:
    positions = atom_positions(result.atoms)
    mass_amu = total_mass_amu(result.atoms)
    print("AA-DPD diagnostics")
    print(f"  atom_count: {len(result.atoms)}")
    print(f"  density_g_cm3: {density_g_cm3(mass_amu, result.box_length_a):.6f}")
    print(f"  box_length_a: {result.box_length_a:.6f}")
    print(f"  converged: {result.converged}")
    print(f"  dpd_steps: {result.steps}")
    print(f"  finite_coords: {bool(np.all(np.isfinite(positions)))}")
    print(f"  min_distinct_atom_distance_a: {min_distinct_distance_a(positions):.6f}")


def import_openmm_deps() -> OpenMMDeps:
    try:
        from mupt.interfaces.rdkit import primitive_to_rdkit_mols
        from openff.interchange import Interchange
        from openff.toolkit import ForceField, Molecule, Topology
        from openff.units import unit as off_unit
        from openmm import LangevinIntegrator, Vec3
        from openmm import unit as omm_unit
        from openmm.app import Simulation
    except ModuleNotFoundError as exc:
        missing = exc.name or "an optional OpenMM validation dependency"
        raise RuntimeError(
            "OpenMM validation dependencies are missing. Install RDKit, "
            "openff-toolkit, openff-interchange, and openmm in this environment, "
            "or rerun with --skip-openmm. "
            f"Missing import: {missing}"
        ) from exc

    return OpenMMDeps(
        Interchange=Interchange,
        ForceField=ForceField,
        Molecule=Molecule,
        Topology=Topology,
        off_unit=off_unit,
        omm_unit=omm_unit,
        LangevinIntegrator=LangevinIntegrator,
        Vec3=Vec3,
        Simulation=Simulation,
        primitive_to_rdkit_mols=primitive_to_rdkit_mols,
    )


def openmm_system_from_interchange(interchange: Any) -> Any:
    for method_name in ("to_openmm_system", "to_openmm"):
        method = getattr(interchange, method_name, None)
        if method is None:
            continue
        try:
            return method(combine_nonbonded_forces=True)
        except TypeError:
            return method()
    raise RuntimeError("OpenFF Interchange has no to_openmm_system() or to_openmm() method.")


def openmm_topology_from_interchange(interchange: Any, topology: Any) -> Any:
    method = getattr(interchange, "to_openmm_topology", None)
    if method is not None:
        return method()
    return topology.to_openmm()


def rdkit_positions_angstrom(rdkit_mols: list[Any]) -> np.ndarray:
    positions = []
    for mol in rdkit_mols:
        conformer = mol.GetConformer()
        for atom_idx in range(mol.GetNumAtoms()):
            position = conformer.GetAtomPosition(atom_idx)
            positions.append((position.x, position.y, position.z))
    return np.asarray(positions, dtype=float)


def energy_kj_mol(simulation: Any, omm_unit: Any) -> float:
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilojoule_per_mole)
    return float(energy)


def run_openmm_validation(root: Any, box_length_a: float, max_iterations: int) -> None:
    deps = import_openmm_deps()
    rdkit_mols = list(
        deps.primitive_to_rdkit_mols(
            root,
            resname_map=PE_RESNAME_MAP,
            default_atom_position=np.zeros(3),
        )
    )
    molecules = [
        deps.Molecule.from_rdkit(
            mol,
            allow_undefined_stereo=True,
            hydrogens_are_explicit=True,
        )
        for mol in rdkit_mols
    ]
    topology = deps.Topology.from_molecules(molecules)
    topology.box_vectors = deps.off_unit.Quantity(np.eye(3) * box_length_a, deps.off_unit.angstrom)
    interchange = deps.Interchange.from_smirnoff(deps.ForceField("openff-2.2.1.offxml"), topology)
    system = openmm_system_from_interchange(interchange)
    openmm_topology = openmm_topology_from_interchange(interchange, topology)

    vectors = (
        deps.Vec3(box_length_a, 0.0, 0.0) * deps.omm_unit.angstrom,
        deps.Vec3(0.0, box_length_a, 0.0) * deps.omm_unit.angstrom,
        deps.Vec3(0.0, 0.0, box_length_a) * deps.omm_unit.angstrom,
    )
    system.setDefaultPeriodicBoxVectors(*vectors)
    if hasattr(openmm_topology, "setPeriodicBoxVectors"):
        openmm_topology.setPeriodicBoxVectors(vectors)

    positions = rdkit_positions_angstrom(rdkit_mols)
    n_openmm_atoms = sum(1 for _ in openmm_topology.atoms())
    if len(positions) != n_openmm_atoms:
        raise RuntimeError(
            f"OpenMM topology atom count ({n_openmm_atoms}) does not match "
            f"RDKit coordinate count ({len(positions)})."
        )

    integrator = deps.LangevinIntegrator(
        300.0 * deps.omm_unit.kelvin,
        1.0 / deps.omm_unit.picosecond,
        0.001 * deps.omm_unit.picoseconds,
    )
    simulation = deps.Simulation(openmm_topology, system, integrator)
    simulation.context.setPositions(positions * deps.omm_unit.angstrom)

    initial_energy = energy_kj_mol(simulation, deps.omm_unit)
    simulation.minimizeEnergy(maxIterations=max_iterations)
    minimized_energy = energy_kj_mol(simulation, deps.omm_unit)
    print("OpenMM diagnostics")
    print(f"  molecule_count: {len(molecules)}")
    print(f"  atom_count: {n_openmm_atoms}")
    print(f"  initial_potential_energy_kj_mol: {initial_energy:.6f}")
    print(f"  minimized_potential_energy_kj_mol: {minimized_energy:.6f}")
    print(f"  finite_energies: {bool(np.isfinite(initial_energy) and np.isfinite(minimized_energy))}")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        root = build_pe_melt(args)
        result = run_dpd(root, args)
        print_dpd_diagnostics(result)
        if args.skip_openmm:
            print("OpenMM diagnostics: skipped (--skip-openmm)")
        else:
            run_openmm_validation(root, result.box_length_a, args.openmm_max_iterations)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
