'''Placement generators based in HOOMD's dissipative particle dynamics (DPD) simulations'''

__author__ = ''

import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

import freud
import gsd, gsd.hoomd 
import hoomd 
from hoomd.write import DCD
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
from scipy.spatial.transform import RigidTransform
from networkx import all_simple_paths

from .base import PlacementGenerator
from ..mutils.iteration import flexible_iterator, sliding_window

from ..geometry.arraytypes import Shape, Dims, N
from ..geometry.measure import normalized
from ..geometry.coordinates.directions import random_unit_vector
from ..geometry.coordinates.reference import origin
from ..geometry.transforms.rigid import rigid_vector_coalignment

from ..mupr.topology import TopologicalStructure
from ..mupr.connection import Connector
from ..mupr.primitives import Primitive, PrimitiveHandle

def pbc(
    positions : np.ndarray[Shape[N, 3], float],
    box : Sequence[float],
) -> np.ndarray[Shape[N, 3], float]:
    '''
    Periodic boundary conditions
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
    minimum_distance : float=0.95,
) -> bool:
    '''
    Check particle separations.
    '''
    positions = snap.particles.position
    box = snap.configuration.box
    aq = freud.locality.AABBQuery(box,positions)
    aq_query = aq.query(
        query_points=positions,
        query_args=dict(r_min=0.0, r_max=minimum_distance, exclude_ii=True),
    )
    nlist = aq_query.toNeighborList()
    if len(nlist)==0:
        LOGGER.info("Inter-particle separation reached.")
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
        density : float=0.2,
        k : float=20000,
        bond_l : float=1.0,
        r_cut : float=1.2,
        kT : float=1.0,
        A : float=5000,
        gamma : float=800,
        dt : float=0.001,
        particle_spacing=1.1,
        step_per_interval : int=1_000,
        report_interval : int=50_000,
        max_steps : int=1_000_000,
    ) -> None:
        # self.primitive = primitive
        self.density = density
        self.k = k
        self.bond_l = bond_l
        self.r_cut = r_cut
        self.kT = kT
        self.A = A
        self.gamma = gamma
        self.dt = dt
        self.particle_spacing = particle_spacing
        
        self.step_per_interval = step_per_interval
        self.report_interval = report_interval
        self.max_steps = max_steps
        
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
        #if any((subprim.shape is None) for subprim in primitive.children):
        #    raise TypeError('Random walk chain builder requires ellipsoidal or spherical beads to determine step sizes')
    
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
        L = np.cbrt(frame.particles.N / self.density) 
        
        # Read info from chains in universe topology into HOOMD Frame
        #frame.bonds.N = self.primitive.topology.number_of_edges()
        #frame.bonds.group = np.zeros((frame.bonds.N,2)) # populate this with bond indices
        bonds : list[tuple[int, int]] = []
        frame.bonds.types = ['a']
        
        particle_indexer : Iterator[int] = count(0)
        h2i : dict[PrimitiveHandle, int] = dict()
        for chain in primitive.topology.chains:
            head_handle, tail_handle = termini = self.get_termini_handles(chain)
            path : list[PrimitiveHandle] = next(all_simple_paths(chain, source=head_handle, target=tail_handle)) # raise StopIteration if no path exists
            for bead_handle in path:
                h2i[bead_handle] = next(particle_indexer)
            
            LOGGER.debug("chain")
            frame.particles.position[h2i[head_handle]] = np.random.uniform( # place head randomly within box bounds
                low=(-L/2),
                high=(L/2),
                size=3,
            )
            for prim_handle_outgoing, prim_handle_incoming in sliding_window(path, 2):
                idx_outgoing, idx_incoming = idx_pair = h2i[prim_handle_outgoing], h2i[prim_handle_incoming]
                LOGGER.debug(f'Adding a bond between "{prim_handle_outgoing}" (idx {idx_outgoing}) and "{prim_handle_incoming}" (idx {idx_incoming})')
                
                bonds.append(idx_pair)
                delta = self.bond_l * random_unit_vector()
                # delta = np.random.uniform(low=(-self.bond_l/2),high=(self.bond_l/2),size=3) #TODO
                # delta /= np.linalg.norm(delta)*self.bond_l
                frame.particles.position[idx_incoming] = frame.particles.position[idx_outgoing] + delta
        
        ## assign bonded index pairs
        frame.bonds.group = bonds
        frame.bonds.N = len(frame.bonds.group)
        
        ## set periodic box based on initial positions and target density
        if (L < 3*self.r_cut):
            L = 3*self.r_cut
            LOGGER.warning(
                f"Small number of particles, lowering density to {frame.particles.N/(L**3)}, and L={L}"
            )
        frame.configuration.box = [L, L, L, 0, 0, 0] # monoclinic cubic box with scale L
        frame.particles.position = pbc(frame.particles.position, [L, L, L])
        
        # set up HOOMD Simulation
        LOGGER.debug('Initializing HOOMD Simulation')
        harmonic = hoomd.md.bond.Harmonic()
        harmonic.params["a"] = dict(r0=self.bond_l, k=self.k)
        
        integrator = hoomd.md.Integrator(dt=self.dt)
        integrator.forces.append(harmonic)
        
        simulation = hoomd.Simulation(device=hoomd.device.auto_select(), seed=np.random.randint(65000))
        simulation.operations.integrator = integrator 
        simulation.create_state_from_snapshot(frame)
        
        const_vol = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator.methods.append(const_vol)
        
        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        simulation.operations.nlist = nlist
        DPD = hoomd.md.pair.DPD(nlist, default_r_cut=self.r_cut, kT=self.kT)
        DPD.params[('A', 'A')] = dict(A=self.A, gamma=self.gamma)
        integrator.forces.append(DPD)
        
        # Run Simulation in intervals until bond lengths converge
        with gsd.hoomd.open(name="dpd.gsd", mode="w") as f:
            f.append(frame)
    
        dcd = DCD(
            trigger=hoomd.trigger.Periodic(self.report_interval),
            filename='dpd_test.dcd',
        )
        simulation.operations.writers.append(dcd)
        
        LOGGER.debug('Beginning HOOMD Simulation')
        # simulation.run(1000)
        snap = simulation.state.get_snapshot()
        total_steps_run : int = 0
        while (not check_inter_particle_distance(snap, minimum_distance=0.95)) and (total_steps_run < self.max_steps):
            if (total_steps_run % self.report_interval) == 0:
                LOGGER.debug(f'Bond lengths not converged after {total_steps_run} steps; continuing simulation')
            simulation.run(self.step_per_interval)
            total_steps_run += self.step_per_interval
        LOGGER.debug('Bond lengths converged; ending simulation')
        snap = simulation.state.get_snapshot()

        # yield placements from final snapshot of simulation
        for handle, idx in h2i.items():
            LOGGER.debug(f'Final position of "{handle}" (idx {idx}): {snap.particles.position[idx]}')
            yield handle, snap.particles.position[idx]
