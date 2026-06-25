#!/usr/bin/env python
"""Manual dense polyethylene AA-DPD/OpenMM smoke-test harness.

Run this from an environment where MuPT is installed. For local development,
install the checkout first with ``pip install -e .``.
"""

from __future__ import annotations

import argparse
import logging
import math
from typing import Any

import numpy as np
import networkx as nx

from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings
from mupt.geometry.coordinates.reference import origin
from mupt.geometry.transforms.rigid import rigid_vector_coalignment
from mupt.interfaces.rdkit import suppress_rdkit_logs
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.mupr.primitives import Primitive
from mupt.mupr.topology import TopologicalStructure
from mupt.roles import assign_SAAMR_roles


LOGGER = logging.getLogger(__name__)

AMU_TO_G = 1.66053906660e-24
ANGSTROM3_TO_CM3 = 1.0e-24
DA_PER_NM3_TO_G_CM3 = 0.00166053906660
PE_SMILES = {
    "head": "[H:1]-[CH2:2]-*",
    "ethane": "*-[CH2:1][CH2:2]-*",
    "tail": "*-[CH2:1]-[H:2]",
}
PE_RESNAME_MAP = {"head": "HEA", "ethane": "EAN", "tail": "TYL"}


def sequence_repeat_units(chain_len: int) -> list[str]:
    """Return a deterministic PE chain sequence including terminal caps."""

    return ["head", *("ethane" for _ in range(chain_len - 2)), "tail"]


def build_pe_lexicon(axis: int = 0) -> dict[str, Primitive]:
    """Build oriented PE repeat-unit primitives from the script SMILES table."""

    lexicon = {}
    with suppress_rdkit_logs():
        for unit_name, smiles in PE_SMILES.items():
            unit = primitive_from_smiles(
                smiles,
                ensure_explicit_Hs=True,
                embed_positions=True,
                label=unit_name,
            )
            head_atom, tail_atom = unit.search_hierarchy_by(
                lambda prim: "molAtomMapNumber" in prim.metadata,
                min_count=2,
            )
            head_pos = head_atom.shape.centroid
            tail_pos = tail_atom.shape.centroid
            major_radius = np.linalg.norm(tail_pos - head_pos) / 2.0
            axis_vec = np.zeros(3, dtype=float)
            axis_vec[axis] = major_radius
            unit.rigidly_transform(
                rigid_vector_coalignment(
                    vector1_start=head_pos,
                    vector1_end=tail_pos,
                    vector2_start=origin(3),
                    vector2_end=axis_vec,
                    t1=0.5,
                    t2=0.0,
                )
            )
            lexicon[unit_name] = unit
    return lexicon


def build_pe_melt_primitive(args: argparse.Namespace) -> Primitive:
    """Build a deterministic all-atom PE primitive without pytest fixtures.

    This function intentionally constructs local residue templates and chain
    topology only. ``AllAtomDPDBuilder`` owns frame-0 chain placement through the
    shared PlacementGenerator abstraction before running HOOMD relaxation.
    """

    np.random.seed(args.seed)
    lexicon = build_pe_lexicon()
    root = Primitive(label="pe_melt")
    for chain_idx in range(args.n_chains):
        segment = Primitive(label=f"chain_{chain_idx:04d}")
        for unit_name in sequence_repeat_units(args.chain_len):
            segment.attach_child(lexicon[unit_name].copy())
        segment.set_topology(
            nx.path_graph(segment.children_by_handle.keys(), create_using=TopologicalStructure),
            max_registration_iter=100,
        )
        root.attach_child(segment)
    assign_SAAMR_roles(root)
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic dense polyethylene melt, run the all-atom "
            "DPD builder with spacing and bonded-energy convergence checks, "
            "and optionally smoke-test OpenMM minimization."
        )
    )
    parser.add_argument("--n-chains", type=int, default=10, help="Number of PE chains to build.")
    parser.add_argument("--chain-len", type=int, default=15, help="Repeat units per PE chain.")
    parser.add_argument("--density-g-cm3", type=float, default=0.85, help="Target melt density.")
    parser.add_argument("--dpd-max-steps", type=int, default=50000, help="Maximum DPD integration steps.")
    parser.add_argument(
        "--particle-spacing-a",
        type=float,
        default=0.75,
        help="Minimum nonbonded atom spacing required for DPD convergence, in Angstrom.",
    )
    parser.add_argument(
        "--dpd-steps-per-interval",
        type=int,
        default=1000,
        help="DPD steps between convergence checks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic build and DPD seed.")
    parser.add_argument("--write-dpd-log", action="store_true", help="Write AA-DPD convergence diagnostics JSONL.")
    parser.add_argument(
        "--dpd-output-name",
        default="pe_melt_aa_dpd",
        help="Output prefix used when --write-dpd-log is enabled.",
    )
    parser.add_argument("--skip-openmm", action="store_true", help="Skip OpenMM minimization smoke test.")
    parser.add_argument(
        "--allow-unconverged-dpd",
        action="store_true",
        help="Do not exit nonzero if the DPD spacing criterion is not reached.",
    )
    parser.add_argument(
        "--min-distance-a",
        type=float,
        default=0.0,
        help="Exit nonzero if the distinct atom minimum distance is at or below this Angstrom threshold.",
    )
    parser.add_argument(
        "--charge-method",
        default="openff-gnn-am1bcc-1.0.0.pt",
        help=(
            "OpenFF partial charge method for the OpenMM minimization smoke test. "
            "Defaults to the NAGL/AshGC model; use zeros/formal_charge only for debug."
        ),
    )
    parser.add_argument("--md-steps", type=int, default=5000, help="Post-minimization NVT MD steps.")
    parser.add_argument("--md-timestep-fs", type=float, default=2.0, help="NVT MD timestep in femtoseconds.")
    parser.add_argument("--md-friction-ps", type=float, default=1.0, help="Langevin friction coefficient in 1/ps.")
    parser.add_argument("--md-temperature-k", type=float, default=300.0, help="NVT MD temperature in kelvin.")
    parser.add_argument("--md-report-interval", type=int, default=500, help="NVT MD diagnostic interval in steps.")
    parser.add_argument("--npt-steps", type=int, default=0, help="Optional post-NVT NPT MD steps.")
    parser.add_argument("--pressure-atm", type=float, default=1.0, help="NPT pressure in atmospheres.")
    parser.add_argument("--barostat-frequency", type=int, default=25, help="Monte Carlo barostat frequency in steps.")
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
    if args.particle_spacing_a <= 0.0:
        raise ValueError("--particle-spacing-a must be > 0")
    if args.min_distance_a < 0.0:
        raise ValueError("--min-distance-a must be >= 0")
    if args.md_steps < 0:
        raise ValueError("--md-steps must be >= 0")
    if args.md_timestep_fs <= 0.0:
        raise ValueError("--md-timestep-fs must be > 0")
    if args.md_friction_ps <= 0.0:
        raise ValueError("--md-friction-ps must be > 0")
    if args.md_temperature_k <= 0.0:
        raise ValueError("--md-temperature-k must be > 0")
    if args.md_report_interval < 1:
        raise ValueError("--md-report-interval must be >= 1")
    if args.npt_steps < 0:
        raise ValueError("--npt-steps must be >= 0")
    if args.pressure_atm <= 0.0:
        raise ValueError("--pressure-atm must be > 0")
    if args.barostat_frequency < 1:
        raise ValueError("--barostat-frequency must be >= 1")


def build_pe_melt(args: argparse.Namespace) -> Any:
    return build_pe_melt_primitive(args)


def run_dpd(root: Any, args: argparse.Namespace) -> Any:
    settings = AllAtomDPDSettings(
        density_g_cm3=args.density_g_cm3,
        n_steps_max=args.dpd_max_steps,
        n_steps_per_interval=args.dpd_steps_per_interval,
        particle_spacing_a=args.particle_spacing_a,
        report_interval=args.dpd_steps_per_interval,
        random_seed=args.seed,
        write_gsd=False,
        write_log=args.write_dpd_log,
        output_name=args.dpd_output_name if args.write_dpd_log else None,
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


def min_distinct_distance_a(positions: np.ndarray, box_length_a: float | None = None) -> float:
    if len(positions) < 2 or not np.all(np.isfinite(positions)):
        return math.nan
    deltas = positions[:, None, :] - positions[None, :, :]
    if box_length_a is not None:
        deltas -= box_length_a * np.round(deltas / box_length_a)
    distances = np.linalg.norm(deltas, axis=-1)
    distances[np.tril_indices(len(positions))] = np.inf
    return float(np.min(distances))


def log_dpd_diagnostics(result: Any) -> tuple[bool, float]:
    positions = atom_positions(result.atoms)
    mass_amu = total_mass_amu(result.atoms)
    finite_coords = bool(np.all(np.isfinite(positions)))
    minimum_distance = min_distinct_distance_a(positions, result.box_length_a)
    LOGGER.info("AA-DPD diagnostics")
    LOGGER.info("  atom_count: %s", len(result.atoms))
    LOGGER.info("  density_g_cm3: %.6f", density_g_cm3(mass_amu, result.box_length_a))
    LOGGER.info("  box_length_a: %.6f", result.box_length_a)
    LOGGER.info("  converged: %s", result.converged)
    LOGGER.info("  dpd_steps: %s", result.steps)
    LOGGER.info("  finite_coords: %s", finite_coords)
    LOGGER.info("  min_periodic_distinct_atom_distance_a: %.6f", minimum_distance)
    diagnostics = getattr(result, "diagnostics", {})
    if diagnostics:
        LOGGER.info("  spacing_converged: %s", diagnostics.get("spacing_converged"))
        LOGGER.info("  bonded_energy_converged: %s", diagnostics.get("bonded_energy_converged"))
        LOGGER.info("  bond_energy_per_term: %s", diagnostics.get("bond_energy_per_term"))
        LOGGER.info("  angle_energy_per_term: %s", diagnostics.get("angle_energy_per_term"))
        LOGGER.info("  dihedral_energy_per_term: %s", diagnostics.get("dihedral_energy_per_term"))
        LOGGER.info("  improper_energy_per_term: %s", diagnostics.get("improper_energy_per_term"))
        LOGGER.info("  dpd_energy: %s", diagnostics.get("dpd_energy"))
    return finite_coords, minimum_distance


def validate_dpd_diagnostics(result: Any, finite_coords: bool, minimum_distance: float, args: argparse.Namespace) -> None:
    if not finite_coords:
        raise RuntimeError("AA-DPD produced nonfinite atom coordinates.")
    if not np.isfinite(minimum_distance):
        raise RuntimeError("AA-DPD minimum atom distance is not finite.")
    if minimum_distance <= args.min_distance_a:
        raise RuntimeError(
            f"AA-DPD minimum atom distance {minimum_distance:.6f} A is not above "
            f"the requested threshold {args.min_distance_a:.6f} A."
        )
    if not args.allow_unconverged_dpd and not result.converged:
        raise RuntimeError("AA-DPD did not satisfy the DPD convergence criterion.")


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


def system_mass_da(system: Any, omm_unit: Any) -> float:
    """Return total OpenMM system mass in daltons."""

    return float(sum(system.getParticleMass(i).value_in_unit(omm_unit.dalton) for i in range(system.getNumParticles())))


def box_density_g_cm3(box_vectors_nm: np.ndarray, mass_da: float) -> float:
    """Return mass density from OpenMM box vectors in nm."""

    volume_nm3 = abs(float(np.linalg.det(box_vectors_nm)))
    return mass_da * DA_PER_NM3_TO_G_CM3 / volume_nm3


def assign_openff_charges(molecules: list[Any], charge_method: str) -> None:
    """Assign OpenFF partial charges, using NAGL explicitly for AshGC models."""

    if charge_method.endswith(".pt") or charge_method.startswith("openff-gnn"):
        from openff.toolkit.utils import ToolkitRegistry
        from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

        if not NAGLToolkitWrapper.is_available():
            raise RuntimeError(
                "NAGL charge assignment requested, but OpenFF NAGL is unavailable. "
                "Install openff-nagl or choose a debug-only method such as zeros."
            )
        registry = ToolkitRegistry([NAGLToolkitWrapper()])
        for molecule in molecules:
            molecule.assign_partial_charges(
                partial_charge_method=charge_method,
                toolkit_registry=registry,
            )
        return

    for molecule in molecules:
        molecule.assign_partial_charges(partial_charge_method=charge_method)


def run_openmm_validation(root: Any, box_length_a: float, charge_method: str, args: argparse.Namespace) -> None:
    from mupt.interfaces.rdkit import primitive_to_rdkit_mols
    from openff.interchange import Interchange
    from openff.toolkit import ForceField, Molecule, Topology
    from openff.units import unit as off_unit
    from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Vec3
    from openmm import unit as omm_unit
    from openmm.app import Simulation

    rdkit_mols = list(
        primitive_to_rdkit_mols(
            root,
            resname_map=PE_RESNAME_MAP,
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
    assign_openff_charges(molecules, charge_method)

    topology = Topology.from_molecules(molecules)
    topology.box_vectors = off_unit.Quantity(np.eye(3) * box_length_a, off_unit.angstrom)
    interchange = ForceField("openff-2.2.1.offxml").create_interchange(
        topology,
        charge_from_molecules=[molecules[0]],
    )
    system = openmm_system_from_interchange(interchange)
    openmm_topology = openmm_topology_from_interchange(interchange, topology)

    vectors = (
        Vec3(box_length_a, 0.0, 0.0) * omm_unit.angstrom,
        Vec3(0.0, box_length_a, 0.0) * omm_unit.angstrom,
        Vec3(0.0, 0.0, box_length_a) * omm_unit.angstrom,
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

    integrator = LangevinMiddleIntegrator(
        args.md_temperature_k * omm_unit.kelvin,
        args.md_friction_ps / omm_unit.picosecond,
        args.md_timestep_fs * omm_unit.femtosecond,
    )
    simulation = Simulation(openmm_topology, system, integrator)
    simulation.context.setPositions(positions * omm_unit.angstrom)

    initial_energy = energy_kj_mol(simulation, omm_unit)
    simulation.minimizeEnergy()
    minimized_energy = energy_kj_mol(simulation, omm_unit)
    LOGGER.info("OpenMM diagnostics")
    LOGGER.info("  molecule_count: %s", len(molecules))
    LOGGER.info("  atom_count: %s", n_openmm_atoms)
    LOGGER.info("  constraints: %s", system.getNumConstraints())
    LOGGER.info("  charge_method: %s", charge_method)
    LOGGER.info("  initial_potential_energy_kj_mol: %.6f", initial_energy)
    LOGGER.info("  minimized_potential_energy_kj_mol: %.6f", minimized_energy)
    LOGGER.info("  finite_energies: %s", bool(np.isfinite(initial_energy) and np.isfinite(minimized_energy)))
    if not (np.isfinite(initial_energy) and np.isfinite(minimized_energy)):
        raise RuntimeError("OpenMM validation produced nonfinite energies.")
    run_nvt_smoke(simulation, system, omm_unit, args)
    run_npt_smoke(
        simulation,
        interchange,
        openmm_topology,
        monte_carlo_barostat_cls=MonteCarloBarostat,
        langevin_middle_integrator_cls=LangevinMiddleIntegrator,
        simulation_cls=Simulation,
        omm_unit=omm_unit,
        args=args,
    )


def run_nvt_smoke(simulation: Any, system: Any, omm_unit: Any, args: argparse.Namespace) -> None:
    """Run a short regular-NVT stability check after minimization."""

    if args.md_steps == 0:
        LOGGER.info("NVT diagnostics: skipped (--md-steps 0)")
        return

    simulation.context.setVelocitiesToTemperature(args.md_temperature_k * omm_unit.kelvin, args.seed)
    LOGGER.info("NVT diagnostics")
    LOGGER.info("  timestep_fs: %.6f", args.md_timestep_fs)
    LOGGER.info("  friction_1_per_ps: %.6f", args.md_friction_ps)
    LOGGER.info("  target_temperature_k: %.6f", args.md_temperature_k)
    LOGGER.info("  requested_steps: %s", args.md_steps)
    steps_run = 0
    while steps_run < args.md_steps:
        steps = min(args.md_report_interval, args.md_steps - steps_run)
        simulation.step(steps)
        steps_run += steps
        state = simulation.context.getState(getEnergy=True)
        potential = float(state.getPotentialEnergy().value_in_unit(omm_unit.kilojoule_per_mole))
        kinetic = float(state.getKineticEnergy().value_in_unit(omm_unit.kilojoule_per_mole))
        finite = bool(np.isfinite(potential) and np.isfinite(kinetic))
        time_ps = steps_run * args.md_timestep_fs / 1000.0
        LOGGER.info(
            "  step %8d time_ps %10.4f potential_kj_mol %14.6f "
            "kinetic_kj_mol %14.6f finite %s",
            steps_run,
            time_ps,
            potential,
            kinetic,
            finite,
        )
        if not finite:
            raise RuntimeError("NVT stability check produced nonfinite energy.")


def run_npt_smoke(
    simulation: Any,
    interchange: Any,
    openmm_topology: Any,
    monte_carlo_barostat_cls: Any,
    langevin_middle_integrator_cls: Any,
    simulation_cls: Any,
    omm_unit: Any,
    args: argparse.Namespace,
) -> None:
    """Continue from the NVT state under regular NPT conditions."""

    if args.npt_steps == 0:
        LOGGER.info("NPT diagnostics: skipped (--npt-steps 0)")
        return

    state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    positions = state.getPositions(asNumpy=True)
    velocities = state.getVelocities(asNumpy=True)
    box_vectors = state.getPeriodicBoxVectors()
    system = openmm_system_from_interchange(interchange)
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    system.addForce(
        monte_carlo_barostat_cls(
            args.pressure_atm * omm_unit.atmosphere,
            args.md_temperature_k * omm_unit.kelvin,
            args.barostat_frequency,
        )
    )
    if hasattr(openmm_topology, "setPeriodicBoxVectors"):
        openmm_topology.setPeriodicBoxVectors(box_vectors)

    integrator = langevin_middle_integrator_cls(
        args.md_temperature_k * omm_unit.kelvin,
        args.md_friction_ps / omm_unit.picosecond,
        args.md_timestep_fs * omm_unit.femtosecond,
    )
    npt = simulation_cls(openmm_topology, system, integrator)
    npt.context.setPositions(positions)
    npt.context.setVelocities(velocities)
    mass_da = system_mass_da(system, omm_unit)

    LOGGER.info("NPT diagnostics")
    LOGGER.info("  pressure_atm: %.6f", args.pressure_atm)
    LOGGER.info("  barostat_frequency: %s", args.barostat_frequency)
    LOGGER.info("  requested_steps: %s", args.npt_steps)
    steps_run = 0
    while steps_run < args.npt_steps:
        steps = min(args.md_report_interval, args.npt_steps - steps_run)
        npt.step(steps)
        steps_run += steps
        state = npt.context.getState(getEnergy=True, enforcePeriodicBox=True)
        potential = float(state.getPotentialEnergy().value_in_unit(omm_unit.kilojoule_per_mole))
        kinetic = float(state.getKineticEnergy().value_in_unit(omm_unit.kilojoule_per_mole))
        box_nm = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(omm_unit.nanometer)
        density = box_density_g_cm3(box_nm, mass_da)
        finite = bool(np.isfinite(potential) and np.isfinite(kinetic) and np.isfinite(density))
        time_ps = steps_run * args.md_timestep_fs / 1000.0
        LOGGER.info(
            "  step %8d time_ps %10.4f potential_kj_mol %14.6f "
            "kinetic_kj_mol %14.6f density_g_cm3 %10.6f finite %s",
            steps_run,
            time_ps,
            potential,
            kinetic,
            density,
            finite,
        )
        if not finite:
            raise RuntimeError("NPT stability check produced nonfinite energy or density.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    try:
        validate_args(args)
        root = build_pe_melt(args)
        result = run_dpd(root, args)
        finite_coords, minimum_distance = log_dpd_diagnostics(result)
        validate_dpd_diagnostics(result, finite_coords, minimum_distance, args)
        if args.skip_openmm:
            LOGGER.info("OpenMM diagnostics: skipped (--skip-openmm)")
        else:
            run_openmm_validation(root, result.box_length_a, args.charge_method, args)
    except Exception as exc:
        LOGGER.error("ERROR: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
