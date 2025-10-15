'''Placement generators based in HOOMD's dissipative particle dynamics (DPD) simulations'''

__author__ = ''

from .base import PlacementGenerator
from mupt.builders.random_walk import AngleConstrainedRandomWalk
import freud
import gsd, gsd.hoomd 
import hoomd 
import numpy as np
import time

def initialize_snapshot_rand_walk(num_pol, num_mon, density=0.85, bond_length=1.0, buffer=0.1):
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
def create_polymer_system_dpd(num_pol,num_mon,density,k=20000,bond_l=1.0,r_cut=1.15,kT=1.0,A=1000,gamma=800,dt=0.001,particle_spacing=1.1):
    
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
    frame = initialize_snapshot_rand_walk(num_mon=num_mon,num_pol=num_pol,bond_length=bond_l,density=density)
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
            return 0
        simulation.run(1000)
        snap=simulation.state.get_snapshot()
        
    end_time = time.perf_counter()
    print("Total build and simulation time:", end_time - start_time)
    return snap.particles.position

