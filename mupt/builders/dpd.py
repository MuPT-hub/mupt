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

def pbc(d, box):
    '''
    Periodic boundary conditions
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

def check_inter_particle_distance(snap, minimum_distance=0.95):
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


class DPD_RandomWalk(PlacementGenerator):
    '''
    Builder which places children of a Primitive
    in a non-self-avoiding random walk and runs a DPD simulation.
    '''
    def __init__(
        self,
        density=0.2,
        k=20000,
        bond_l=1.0,
        r_cut=1.2,
        kT=1.0,
        A=5000,
        gamma=800,
        dt=0.001,
        particle_spacing=1.1
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
        
    # optional helper methods (to declutter casework from main logic)
    def get_termini_handles(self, chain : TopologicalStructure) -> tuple[Hashable, Hashable]:
        '''
        Find the terminal node(s) of what is assumed to be a linear (path) graph
        Returns the pair of node labels of the termini (a pair of the same value twice for single-node graphs)
        '''
        termini = tuple(chain.termini)
        print(termini) #DEBUG
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
        frame = gsd.hoomd.Frame()
        frame.particles.types = ['A'] #TODO: introduce HMT's?
        frame.particles.N = primitive.topology.number_of_nodes() 
        frame.particles.typeid = np.zeros(frame.particles.N)
        frame.particles.position = np.zeros((frame.particles.N,3)) #populate with random walks
        #frame.bonds.N = self.primitive.topology.number_of_edges()
        #frame.bonds.group = np.zeros((frame.bonds.N,2)) #populate this with bond indices
        bonds = []
        frame.bonds.types = ['a']
        L = np.cbrt(frame.particles.N / self.density) 
        if (L<3*self.r_cut):
            L = 3*self.r_cut
            print("Warning: Small number of particles, lowering density to {}, and L={}".format(frame.particles.N/(L**3),L))
        frame.configuration.box = [L, L, L, 0, 0, 0]

        h2i = {}
        i=0
        for chain in primitive.topology.chains:
            head_handle, tail_handle = termini = self.get_termini_handles(chain) #TODO: Chains missing edges?
            path : list[PrimitiveHandle] = next(all_simple_paths(chain, source=head_handle, target=tail_handle)) # raise StopIteration if no path exists
            h2i[head_handle] = i
            frame.particles.position[i] = np.random.uniform(low=(-L/2),high=(L/2),size=3)
            print("chain")
            for prim_handle_outgoing, prim_handle_incoming in sliding_window(path, 2):
                print("adding a bond")
                i+=1
                h2i[prim_handle_incoming] = i        
                bonds.append( [h2i[prim_handle_outgoing],h2i[prim_handle_incoming]] )
                delta = np.random.uniform(low=(-self.bond_l/2),high=(self.bond_l/2),size=3) #TODO
                delta /= np.linalg.norm(delta)*self.bond_l
                frame.particles.position[h2i[prim_handle_incoming]] = frame.particles.position[h2i[prim_handle_outgoing]] + delta
        
        frame.particles.position = pbc(frame.particles.position,[L,L,L])
        frame.bonds.group = bonds
        frame.bonds.N = len(frame.bonds.group)
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
        
        simulation.run(1000)
        snap=simulation.state.get_snapshot()
        while not check_inter_particle_distance(snap,minimum_distance=0.95): #TODO: update min_distance?
            simulation.run(1000)
        snap=simulation.state.get_snapshot()

        for chain in primitive.topology.chains:
            head_handle, tail_handle = termini = self.get_termini_handles(chain) #TODO: Same issue as above
            path : list[PrimitiveHandle] = next(all_simple_paths(chain, source=head_handle, target=tail_handle)) # raise StopIteration if no path exists
            for handle in path:
                yield handle, snap.particles.position[h2i[handle]]
