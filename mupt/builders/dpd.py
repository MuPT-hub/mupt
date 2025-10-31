'''Placement generators based in HOOMD's dissipative particle dynamics (DPD) simulations'''

__author__ = ''

import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

import freud
import gsd, gsd.hoomd 
import time
import hoomd 
from hoomd.write import DCD
from hoomd.write import GSD
from hoomd.trigger import Periodic

from typing import (
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Optional,
    Sized,
    Union,
    Sequence,
)
from numbers import Number
from collections import defaultdict
from itertools import count

import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation
from networkx import all_simple_paths

from .base import PlacementGenerator
from ..mutils.iteration import flexible_iterator, sliding_window

from ..geometry.arraytypes import Shape, Dims, N
from ..geometry.measure import normalized
from ..geometry.coordinates.directions import random_unit_vector
from ..geometry.coordinates.reference import origin
from ..geometry.transforms.rigid import rigid_vector_coalignment
from ..geometry.shapes import Sphere, Ellipsoid

from ..mupr.topology import TopologicalStructure
from ..mupr.connection import Connector, TraversalDirection
from ..mupr.primitives import Primitive, PrimitiveHandle

def pbc(
    positions : np.ndarray[Shape[N, 3], float],
    box : Sequence[float],
) -> np.ndarray[Shape[N, 3], float]:
    '''
    Apply periodic boundary conditions to a set of positions,
    "wrapping" them into the box defined by "box"
    '''
    for i in range(3):
        a = positions[:,i]
        pos_max = np.max(a)
        pos_min = np.min(a)
        while pos_max > box[i]/2 or pos_min < -box[i]/2:
            a[a < -box[i]/2] += box[i]
            a[a >  box[i]/2] -= box[i]
            pos_max = np.max(a)
            pos_min = np.min(a)
    return positions # TB: is "a" acted on in-place here? If so, why return the array that's already been modified in-place?

def check_inter_particle_distance(
    snap : hoomd.Snapshot,
    minimum_distance : float=1.05,
) -> bool:
    '''
    Return whether all distinct particles in the snapshot 
    have separated to a distance greater than minimum_distance
    '''
    positions = snap.particles.position
    box = snap.configuration.box
    aq = freud.locality.AABBQuery(box,positions)
    aq_query = aq.query(
        query_points=positions,
        query_args=dict(
            r_min=0.0,
            r_max=minimum_distance,
            exclude_ii=True,
        ),
    )
    nlist = aq_query.toNeighborList()
    
    if len(nlist) == 0:
        LOGGER.info(f'Inter-particle separation >{minimum_distance} reached')
        return True
    else:
        return False


class DPDRandomWalk(PlacementGenerator):
    '''
    Builder which places children of a Primitive
    in a non-self-avoiding random walk and runs a DPD simulation.
    '''
    def __init__(
        self,
        density : float=0.8,
        k : float=20000,
        bond_length : float=1.1,
        r_cut : float=1.2,
        kT : float=1.0,
        A : float=5000,
        gamma : float=800,
        dt : float=0.001,
        particle_spacing : float=1.0,
        bead_separation : float=0.0,
        n_steps_per_interval : int=1_000,
        n_steps_max : int=1_000_000,
        report_interval : int=1_000,
        output_name : Optional[str]=None,
    ) -> None:
        '''
        Parameters
        ----------
        density : float
            Target number density of particles (representing beads) in DPD simulation
        k : float
            Bond spring constant
        bond_length : float
            Equilibrium bond length
        r_cut : float
            <seek clarification>
        kT : float
            Thermostat temperature, in energy units
        A : float
            Maximum pairwise repulsion force between particles
        gamma : float
            Effective friction coefficient (in units is velocity)
        dt : float
            Time step for simulation
        particle_spacing : float
            Lower bound on particle separation to consider the system converged
        bead_separation : float
            The "real" (i.e. unscaled) distance in angstrom between adjacent beads
            Represents a bond length between bead anchor points
            
        n_steps_per_interval : int
            Number of simulation steps to run between convergence checks
        n_steps_max : int
            Maximum number of simulation steps to run before returning (regardless of convergence)
        report_interval : int
            Number of steps between debug logging reports during simulation
        output_name : Optional[str]
            Filename for initial snapshot and trajectory output, if those are desired
            No output produced if None
        '''
        self.density = density
        self.k = k
        self.bond_length = bond_length
        self.r_cut = r_cut
        self.kT = kT
        self.A = A
        self.gamma = gamma
        self.dt = dt
        self.particle_spacing = particle_spacing
        
        self.bead_separation = bead_separation
        self.n_steps_per_interval = n_steps_per_interval
        self.n_steps_max = n_steps_max
        self.report_interval = report_interval
        self.output_name = output_name
        
    # optional helper methods (to declutter casework from main logic)
    def get_termini_handles(self, chain : TopologicalStructure) -> tuple[Hashable, Hashable]:
        '''
        Find the terminal node(s) of what is assumed to be a linear (path) graph
        Returns the pair of node labels of the termini (a pair of the same value twice for single-node graphs)
        '''
        termini = tuple(chain.termini)
        LOGGER.debug(termini)
        if len(termini) == 2:
            return termini
        elif len(termini) == 1: 
            return termini[0], termini[0]
        else:
            raise ValueError('Unbranched topology must have either 1 or 2 terminal nodes')

    # implementing builder contracts
    def check_preconditions(self, primitive : Primitive) -> None:
        '''Enforce that no branches chains exist anywhere'''
        if primitive.topology.is_branched:
            raise ValueError('Random walk chain builder behavior undefined for branched topologies')
        
        #TODO: Add shapes
        for subprim in primitive.children:
            if not isinstance(subprim.shape, (Ellipsoid, Sphere)):
                raise ValueError('Random walk chain builder requires ellipsoidal or spherical beads to determine step sizes')
    
    def _generate_placements(self, primitive : Primitive) -> Generator[tuple[PrimitiveHandle, np.ndarray], None, None]:
        '''
        Trying to use universe of chains to set monomer positions
        primitive passed in here should be a universe primitive that has chains to loop over 
        paths are lists of handles
        If we assume chains are looped over in the same way, we can map from handles to indices
        '''
        # Initialize HOOMD Frame (initial snapshot) and periodic box
        frame = gsd.hoomd.Frame()
        
        ## Pre-allocate space for particles
        frame.particles.types = ['A'] # TODO: introduce HMT's?
        frame.particles.N = primitive.topology.number_of_nodes() # TB: would be nice to set after iterating over children, but needed to size box
        frame.particles.typeid = np.zeros(frame.particles.N)
        frame.particles.position = np.zeros((frame.particles.N, 3)) # populate with random walks

        ## size (for now cubic) periodic box
        L = np.cbrt(frame.particles.N / self.density) 
        if (L < 3*self.r_cut):
            L : float = 3*self.r_cut
            V_new : float = L**3
            LOGGER.warning(
                f"Small number of particles, lowering density to {frame.particles.N / V_new}, and L={L}"
            )
        
        # Read info from chains in universe topology into HOOMD Frame
        #frame.bonds.N = self.primitive.topology.number_of_edges()
        #frame.bonds.group = np.zeros((frame.bonds.N,2)) # populate this with bond indices
        bonds : list[tuple[int, int]] = []
        bond_types : list[str] = ['a']
        
        hoomd_chains : dict[int, tuple[int]] = dict() # for preserving chain order for orientation calc
        handle_to_particle_idx : dict[PrimitiveHandle, int] = dict()
        reference_anchor_positions : dict[int, np.ndarray[Shape[2, 3], float]] = dict()
        effective_radii : dict[int, float] = dict()
        
        particle_indexer : Iterator[int] = count(0)
        for chain_idx, chain in enumerate(primitive.topology.chains):
            head_handle, tail_handle = termini = self.get_termini_handles(chain)
            path : list[PrimitiveHandle] = next(all_simple_paths(chain, source=head_handle, target=tail_handle)) # raise StopIteration if no path exists

            chain_indices : list[int] = []
            for bead_handle in path:
                is_terminal : bool = ((bead_handle == head_handle) or (bead_handle == tail_handle))
                # determine unique int idx for corresponding particle in HOOMD Frame
                particle_idx = next(particle_indexer)
                chain_indices.append(particle_idx)
                handle_to_particle_idx[bead_handle] = particle_idx

                # determine reference anchor points for effective radius scaling and orientation back-calculation post-simulation
                anchor_positions = np.zeros((2, 3), dtype=float)
                bead_prim : Primitive = primitive.fetch_child(bead_handle)
                for conn_handle, conn in bead_prim.connectors.items():
                    traver_dir : TraversalDirection = next(att for att in conn.anchor.attachables if isinstance(att, TraversalDirection))
                    traver_dir_idx : dict[TraversalDirection, int] = {
                        TraversalDirection.ANTERO: 0,
                        TraversalDirection.RETRO: 1,
                    }
                    anchor_positions[traver_dir_idx[traver_dir],:] = conn.anchor.position
                    if is_terminal:
                        radial_vector = conn.anchor.position - bead_prim.shape.centroid
                        diametric_anchor_pos = bead_prim.shape.centroid - radial_vector
                        anchor_positions[traver_dir_idx[TraversalDirection.complement(traver_dir)],:] = diametric_anchor_pos 
                reference_anchor_positions[particle_idx] = anchor_positions

                r_eff : float = np.linalg.norm(np.subtract(*anchor_positions)) / 2.0 
                effective_radii[particle_idx] = r_eff
            hoomd_chains[chain_idx] = tuple(chain_indices)
            
            # assign positions to LJ particle counterparts in simulation
            frame.particles.position[handle_to_particle_idx[head_handle]] = np.random.uniform( # place head randomly within box bounds
                low=(-L/2),
                high=(L/2),
                size=3,
            )
            for prim_handle_outgoing, prim_handle_incoming in sliding_window(path, 2):
                idx_outgoing, idx_incoming = idx_pair = handle_to_particle_idx[prim_handle_outgoing], handle_to_particle_idx[prim_handle_incoming]
                # LOGGER.debug(f'Adding a bond between "{prim_handle_outgoing}" (idx {idx_outgoing}) and "{prim_handle_incoming}" (idx {idx_incoming})')
                bonds.append(idx_pair)
                
                delta = self.bond_length * random_unit_vector()
                frame.particles.position[idx_incoming] = frame.particles.position[idx_outgoing] + delta
        
        # Specify system for HOOMD Simulation
        ## define integrator
        integrator = hoomd.md.Integrator(dt=self.dt)
        const_vol = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator.methods.append(const_vol)
        LOGGER.debug(f'Defined constant-volume integrator with time step={self.dt}')
        
        ## assign bonded index pairs and bond parameters
        frame.bonds.group = bonds
        frame.bonds.N = len(bonds)
        LOGGER.debug(f'Assigned {frame.bonds.N} bonded pairs to HOOMD topology')
        
        frame.bonds.types = bond_types
        for bond_type in bond_types:
            harmonic = hoomd.md.bond.Harmonic()
            harmonic.params[bond_type] = dict(r0=self.bond_length, k=self.k)
            integrator.forces.append(harmonic)
            LOGGER.debug(f'Set harmonic bond parameters for bond type "{bond_type}": r0={self.bond_length}, k={self.k}')
        
        ## set periodic box based on initial positions and target density
        R_max = max(effective_radii.values()) # for scaling out of LJ units at the end
        frame.configuration.box = [L, L, L, 0, 0, 0] # monoclinic cubic box with scale L
        frame.particles.position = pbc(frame.particles.position, [L, L, L])
        
        # Initialize HOOMD Simulation
        LOGGER.info('Initializing HOOMD Simulation')
        simulation = hoomd.Simulation(
            device=hoomd.device.auto_select(),
            seed=np.random.randint(65_000),
        )
        simulation.operations.integrator = integrator 
        simulation.create_state_from_snapshot(frame)
        
        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        simulation.operations.nlist = nlist
        DPD = hoomd.md.pair.DPD(nlist, default_r_cut=self.r_cut, kT=self.kT)
        DPD.params[('A', 'A')] = dict(A=self.A, gamma=self.gamma)
        integrator.forces.append(DPD)
        
        # Run Simulation in intervals until bond lengths converge
        if (self.output_name is not None):
            with gsd.hoomd.open(name=f'{self.output_name}_init.gsd', mode='w') as f:
                f.append(frame)
            gsd1 = GSD(
                trigger=hoomd.trigger.Periodic(self.report_interval),
                filename=f'{self.output_name}_traj.gsd',
            )
            simulation.operations.writers.append(gsd1)
        
        LOGGER.info('Beginning HOOMD Simulation Run')
        simulation.run(1)
        hoomd_time = time.perf_counter()
        total_steps_run : int = 0
        while not check_inter_particle_distance(
            snap=simulation.state.get_snapshot(),
            minimum_distance=self.particle_spacing,
        ):
            simulation.run(self.n_steps_per_interval)
            total_steps_run += self.n_steps_per_interval
            if (total_steps_run % self.report_interval) == 0:
                LOGGER.debug(f'Integrated {total_steps_run} steps; continuing simulation')
            
            if (total_steps_run >= self.n_steps_max):
                LOGGER.warning(f'Some particles are still too close after maximum simulation step {self.n_steps_max} reached; terminating simulation early')
                break
        end_time = time.perf_counter()
        LOGGER.info(f'HOOMD simulation concluded after {total_steps_run} steps ({end_time - hoomd_time}s walltime)')

        # apply proper scaling to LJ beads and post-process final snapshot
        ## determine on-body (assumed spherical) secant points for each LJ sphere
        snap = simulation.state.get_snapshot()
        scale_factor : float = 2*R_max + self.bead_separation # NOTE: scaling by max ensures beads never intersect, even with 0 bead separation
        positions_scaled = scale_factor * snap.particles.position

        orient_marker_points = np.zeros((frame.particles.N, 3, 3), dtype=float) # each 3x3 slice store incoming-center-outgoing pos for particles
        for chain_idx, particle_indices in hoomd_chains.items():
            chain_particle_centers = positions_scaled[particle_indices,:]
            chain_radii = np.array([effective_radii[idx] for idx in particle_indices]) # shape[N]

            ## determine steps to secant points on spheres forward and backward along chain relative to bead centers
            unit_step_vectors = normalized( np.diff(chain_particle_centers, axis=0) ) # shape [N - 1]
            fwd_steps =  chain_radii[:-1, np.newaxis] * unit_step_vectors  
            bwd_steps = -chain_radii[ 1:, np.newaxis] * unit_step_vectors
            fwd_steps = np.vstack([fwd_steps, -bwd_steps[-1]]) # final step would "step past" the tail bead by same amount as incoming into tail (but in opposite direction)
            bwd_steps = np.vstack([-fwd_steps[-1], bwd_steps]) # first step would "step before" the head bead by same amount as outgoing from head (but in opposite direction)

            ## take steps to set incoming and outgoing positions for all beads
            orient_marker_points[particle_indices, 0, :] = chain_particle_centers + fwd_steps
            orient_marker_points[particle_indices, 1, :] = chain_particle_centers
            orient_marker_points[particle_indices, 2, :] = chain_particle_centers + bwd_steps
            # LOGGER.debug(f'Chain #{chain_idx} has markers {orient_marker_points[particle_indices,:,:]}')

        ## determine and cache final PBC unit cell parameters
        Lx, Ly, Lz, alpha, beta, gamma = snap.configuration.box
        box_scaled = [
            float(scale_factor*Lx), # coerce from numpy float for eventual SD file storage
            float(scale_factor*Ly),
            float(scale_factor*Lz),
            alpha,
            beta,
            gamma,
        ]
        primitive.metadata['unit_cell_parameters'] = box_scaled
        LOGGER.info(f'Final box: {box_scaled}') # TODO: scale up box

        # yield placements from final snapshot of simulation
        for handle, particle_idx in handle_to_particle_idx.items():
            point_incoming, point_center, point_outgoing = orient_marker_points[particle_idx, :, :]
            reference_incoming, reference_outgoing = reference_anchor_positions[particle_idx]

            secant_vector = point_outgoing - point_incoming # span chord within spehrical bead between inter-bead anchor points
            placement = rigid_vector_coalignment(
                reference_incoming,
                reference_outgoing,
                point_center,
                point_center + secant_vector, # radial vector parallel to the secant
                t1=0.5, # align midpoint of reference anchors...
                t2=0.0, # with center of final bead position
            )
            yield handle, placement
