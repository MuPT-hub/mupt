'''Placement generators based in HOOMD's dissipative particle dynamics (DPD) simulations'''

__author__ = ''

import freud
import gsd, gsd.hoomd 
import hoomd 
import numpy as np
import time

from typing import (
    Generator,
    Hashable,
    Iterable,
    Optional,
    Sized,
    Union,
)
from numbers import Number
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import RigidTransform
from networkx import all_simple_paths

from .base import PlacementGenerator
from ..mutils.iteration import flexible_iterator, sliding_window

from ..geometry.arraytypes import Shape, Dims
from ..geometry.measure import normalized
from ..geometry.coordinates.directions import random_unit_vector
from ..geometry.coordinates.reference import origin
from mupt.geometry.transforms.rigid import rigid_vector_coalignment

from ..mupr.topology import TopologicalStructure
from ..mupr.connection import Connector
from ..mupr.primitives import Primitive, PrimitiveHandle

def initialize_snapshot_rand_walk(num_pol, num_mon, density=0.85, bond_length=1.0, buffer=0.1): #this function is being replaced, leaving now for reference
    '''
    Create a HOOMD snapshot of a cubic box with the number density given by input parameters.
    Configure particles using a naiive random walk.
    
    '''    
    N = num_pol * num_mon
    L = np.cbrt(N / density)  # Calculate box size based on density
    positions = np.zeros((N, 3))
    for i in range(num_pol):
        start = i * num_mon
        positions[start] = np.random.uniform(low=(-L/2),high=(L/2),size=3)
        for j in range(num_mon - 1):
            delta = np.random.uniform(low=(-bond_length/2),high=(bond_length/2),size=3)
            delta /= np.linalg.norm(delta)*bond_length
            positions[start+j+1] = positions[start+j] + delta
    positions = pbc(positions,[L,L,L])
    bonds = []
    for i in range(num_pol):
        start = i * num_mon
        for j in range(num_mon - 1):
            bonds.append([start + j, start + j + 1])
    bonds = np.array(bonds)
    frame = gsd.hoomd.Frame()
    frame.particles.types = ['A']
    frame.particles.N = N
    frame.particles.position = positions
    frame.bonds.N = len(bonds)
    frame.bonds.group = bonds
    frame.bonds.types = ['b']
    frame.configuration.box = [L, L, L, 0, 0, 0]
    return frame

def pbc(d,box):
    '''
    periodic boundary conditions
    
    '''
    for i in range(3):
        a = d[:,i]
        pos_max = np.max(a)
        pos_min = np.min(a)
        while pos_max > box[i]/2 or pos_min < -box[i]/2:
            a[a < -box[i]/2] += box[i]
            a[a >  box[i]/2] -= box[i]
            pos_max = np.max(a)
            pos_min = np.min(a)
    return d

def check_inter_particle_distance(snap,minimum_distance=0.95):
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
        print("Inter-particle separation reached.")
        return True
    else:
        return False
def create_polymer_system_dpd(num_pol,num_mon,positions,bonds,density,k=20000,bond_l=1.0,r_cut=1.15,kT=1.0,A=1000,gamma=800,dt=0.001,particle_spacing=1.1):
    
    '''
    Initialize a polymer system in a cubic box using a random walk and a HOOMD simulation with DPD forces.

    ----------
    Parameters
    ----------
    num_pol : int, required
        number of polymers in system
    num_mon : int, required
        length of polymers in system
    density : float, required
        number density to initalize the system
    k : int, default 20000
        spring constant for harmonic bonds
    bond_l : float, default 1.0
        harmonic bond rest length
    r_cut : float, default 1.15
        cutoff pair distance for neighbor list
    kT : float, default 1.0
        temperature of thermostat
    A : float, default 1000
        DPD force parameter
    gamma : float, default 800
        DPD drag parameter (mass/time)
    dt : float, default 0.001
        timestep for HOOMD simulation
    particle_spacing : float, default 1.1
        condition for ending the soft push simulation

    -------
    Returns
    -------
    
    positions : list
        returns list of particle positions
        
    '''
    print(num_pol*num_mon)
    print(f"\nRunning with A={A}, gamma={gamma}, k={k}, "
          f"num_pol={num_pol}, num_mon={num_mon}")
    start_time = time.perf_counter()
    N = num_pol * num_mon
    L = np.cbrt(N / density)
    frame = gsd.hoomd.Frame()
    frame.particles.types = ['A']
    frame.particles.N = N
    frame.particles.position = positions
    frame.bonds.N = len(bonds)
    frame.bonds.group = bonds
    frame.bonds.types = ['b']
    frame.configuration.box = [L, L, L, 0, 0, 0]
    build_stop = time.perf_counter()
    print("Total build time: ", build_stop-start_time)
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params["b"] = dict(r0=bond_l, k=k)
    integrator = hoomd.md.Integrator(dt=dt)
    integrator.forces.append(harmonic)
    simulation = hoomd.Simulation(device=hoomd.device.auto_select(), seed=np.random.randint(65535))# TODO seed
    simulation.operations.integrator = integrator 
    simulation.create_state_from_snapshot(frame)
    const_vol = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(const_vol)
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    simulation.operations.nlist = nlist
    DPD = hoomd.md.pair.DPD(nlist, default_r_cut=r_cut, kT=kT)
    DPD.params[('A', 'A')] = dict(A=A, gamma=gamma)
    integrator.forces.append(DPD)
    
    simulation.run(0)
    simulation.run(1000)
    snap=simulation.state.get_snapshot()
    N = num_pol*num_mon
    time_factor = N/9000
 
    while not check_inter_particle_distance(snap,minimum_distance=0.95):
        check_time = time.perf_counter()
        if (check_time-start_time) > 60*time_factor:
            yield 0
        simulation.run(1000)
        snap=simulation.state.get_snapshot()
        
    end_time = time.perf_counter()
    print("Total build and simulation time:", end_time - start_time)
    return snap.particles.position

class DPD_RandomWalk(PlacementGenerator):
    '''
    Builder which places children of a Primitive
    in a non-self-avoiding random walk and runs a DPD simulation.
    '''
    def __init__(
        self,
        num_pol,
        num_mon,
        density,
        k=20000,
        bond_l=1.0,
        r_cut=1.15,
        kT=1.0,
        A=1000,
        gamma=800,
        dt=0.001,
        particle_spacing=1.1
    ) -> None:
        self.num_pol = num_pol
        self.num_mon = num_mon
        self.density = density
        self.k = k
        self.bond_l = bond_l
        self.r_cut = r_cut
        self.kT = kT
        self.A = A
        self.gamma = gamma
        self.dt = dt
        self.particle_spacing = particle_spacing

    # optional helper methods (to declutter casework from main logic)
    def get_termini_handles(self, chain : TopologicalStructure) -> tuple[Hashable, Hashable]:
        '''
        Find the terminal node(s) of what is assumed to be a linear (path) graph
        Returns the pair of node labels of the termini (a pair of the same value twice for single-node graphs)
        '''
        termini = tuple(chain.termini)
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
        
        if any((subprim.shape is None) for subprim in primitive.children):
            raise TypeError('Random walk chain builder requires ellipsoidal of spherical beads to determine step sizes')
    
    def _generate_placements(self, primitive : Primitive) -> Generator[tuple[PrimitiveHandle, RigidTransform], None, None]:
        '''
        Reorient bodies to be coincident (along a predefined axis) with 
        the steps of an angle-constrained non-self-avoiding random walk
        '''

        center_positions : np.ndarray = create_polymer_system_dpd( #move
            num_pol=self.num_pol,
            num_mon=self.num_mon,
            density=self.density,
            k=self.k,
            bond_l=self.bond_l,
            r_cut=self.r_cut,
            kT=self.kT,
            A=self.A,
            gamma=self.gamma,
            dt=self.dt,
            particle_spacing=self.particle_spacing)

        
        for chain in primitive.topology.chains: #use this to define bonds
            #pull edges out of graph for bonds
            #chain.edges
            # DEV: taking extra care to ensure chain is oriented from end-to-end, because there's no requirement
            # (or indeed, reason to believe) that the order of nodes in chain.nodes is meaningful
            head_handle, tail_handle = termini = self.get_termini_handles(chain)
            path : list[PrimitiveHandle] = next(all_simple_paths(chain, source=head_handle, target=tail_handle)) # raise StopIteration if no path exists
            
            # determine pair of anchor points per-body that alignment is based upon
            connection_points : dict[PrimitiveHandle, list[np.ndarray, np.ndarray]] = defaultdict(list)
            connection_points[head_handle].append(primitive.children_by_handle[head_handle].shape.centroid)
            for prim_handle_outgoing, prim_handle_incoming in sliding_window(path, 2):
                conn_handle_outgoing, conn_handle_incoming = primitive.internal_connection_between(
                    from_child_handle=prim_handle_outgoing,
                    to_child_handle=prim_handle_incoming,
                )
                # NOTE: traversal in-path-order is what guarantees these appends place everything in the correct order
                conn_outgoing = primitive.fetch_connector_on_child(prim_handle_outgoing, conn_handle_outgoing)
                connection_points[prim_handle_outgoing].append(conn_outgoing.anchor_position) # will raise Exception is anchor position is unset
                
                conn_incoming = primitive.fetch_connector_on_child(prim_handle_incoming, conn_handle_incoming)
                connection_points[prim_handle_incoming].append(conn_incoming.anchor_position) # will raise Exception is anchor position is unset
                
                Connector.mutually_antialign_ballistically(conn_outgoing, conn_incoming) # align linkers w/ other's anchor while leaving anchors themselves undisturbed
            # NOTE: order is critical here; only placing tail point AFTER its incoming connection point is inserted
            connection_points[tail_handle].append(primitive.children_by_handle[tail_handle].shape.centroid)
            
            ## extract step sizes from conntions point - NOTE: by design, makes no reference to the shape of the body
            step_sizes : list[float] = []
            for handle in path: # NOTE: iterating over path (rather than connection_points.items()) to guarantee traversal order
                conn_start, conn_end = connection_points[handle]
                step_sizes.append( np.linalg.norm(conn_end - conn_start) + self.bond_length ) # step longer to account for target bond length
            
            # generate random walk steps and corresponding placements
            rw_steps : Generator[np.ndarray, None, None] = random_walk_jointed_chain(
                step_size=step_sizes,
                n_steps_max=len(path), # not strictly necessary, but suppresses "indeterminate num steps" warnings
                initial_point=self.initial_point,
                initial_direction=self.initial_direction,
                clip_angle=self.angle_max_rad,
                dimension=3,
            )
            for handle, (step_start, step_end) in zip(path, sliding_window(rw_steps, 2)):
                LOGGER.debug(f'Random walk placing body {handle} along vector from {step_start} to {step_end}')
                conn_start, conn_end = connection_points[handle]
                t_body = 0.5 # NOTE: no need for special case at termini, since the step size matches the half-body (e.g. center-to-anchor) step size
                
                full_step_len = np.linalg.norm(step_end - step_start)
                step_correction = 1 / (1 + (self.bond_length / full_step_len)) # scale back to account for bond length being included in step size
                t_step = 0.5 * step_correction # adjust step fraction to account for bond length (will always be strictly smaller than 0.5, since ratio of lengths is positive)
                
                placement_transform = rigid_vector_coalignment(
                    # vector 1: spans between anchors of connection point on body
                    conn_start,
                    conn_end,
                    # vector 2: spans between consecutive random walk steps
                    step_start,
                    step_end,
                    # interpolation parameters for which point on respective vectors will be forced exactly-coexistent
                    t1=t_body, # take midpoint (or end, if at termini) of body-anchoring vector
                    t2=t_step, # ...to midpoint of random walk step vector
                )
                #yield handle, placement_transform
                #import random walk code instead of copying
                #take density, and apply pbc on placement_transform coordinates
                #pass in placement_transforms and bonds into DPD code

