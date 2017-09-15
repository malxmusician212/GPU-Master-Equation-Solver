## This files contains all functions necessary to evolve an open quantum system in time.  In particular, one can use the functions here to advance the Lindblad Master Equation PDE with the Runge-Kutta 4 numerical scheme (where space is discretized).

## In order to run a simulation, the user need only call the runrk4_mesolve function.

from __future__ import division
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import skcuda.linalg as linalg
import skcuda

class GPU_mesolve_internals(object):
    
    ## This class initializes all the necessary parameters in the __init__ function below.
    
    def __init__(self, dt_cpu, hamiltonian_cpu, rho_initial_cpu, e_ops, diss_ops, tlist, tol, adapt, coupling_consts):
        
        ## dt_cpu is the initial timestep.  If adaptive timestep is used, then the initial timestep will be  (tlist[0] - tlist[1])/2; if adaptive timestep is not used, then dt_cpu is 0.001 by default.
        ## hamiltonian_cpu is the Hamiltonian of the system, which is assumed to have no time dependence.
        ## rho_initial_cpu is the initial condition for the density matrix, commonly written as rho.
        ## e_ops are the evaluation operators, specified by the user.  When the simulation is finished, the user will only receive the expectation values of the evaluation operators.
        ## diss_ops are the dissipation operators, also specified by the user.
        ## tlist is a list, specified by the user.  The expectation values of the operators in e_ops will be returned for times in the tlist.
        ## tol is the error tolerance, which is needed if an adaptive timestep is active.
        ## adapt is a binary switch: 0 means that adaptive timestep is *not* used, whereas 1 means that adaptive timestep is used.
        ## coupling_consts are the coupling constants, commonly referred to as gamma_i, in the Lindblad Master Equation.
        
        self.dimension = hamiltonian_cpu.shape[0] # dimension is 2^n where n is the number of qubits
        self.hamiltonian = gpuarray.to_gpu(np.asarray(hamiltonian_cpu).astype(np.complex_)) # copy the Hamiltoninan to the GPU 
        
        self.dt_original = np.asarray(dt_cpu) # make sure that the timestep is an array
        self.dt_dummy = self.dt_original # this is used only when we need to increase or decrease our timestep for a single computation
        self.dt_full = gpuarray.to_gpu(np.asarray([self.dt_dummy], dtype=np.complex_)) # a full and a half timestep is needed if adaptive timestep is used
        self.dt_half = 0.5*self.dt_full

        self.rho_evolved_cpu = np.asarray(rho_initial_cpu).astype(np.complex_)
        self.rho_evolved = gpuarray.to_gpu(self.rho_evolved_cpu) # create array of zeros on the GPU to store time evolutions of the density matrix
        self.rho_evolved_for_adaptive = gpuarray.to_gpu(self.rho_evolved_cpu)
        self.rho_storage = gpuarray.zeros_like(self.rho_evolved)
        
        self.dissipation_ops = []
        self.dissipation_ops_herm = []
        self.couple_consts = []
        self.const_lindblad_term = []
        
        for ii in range(0,len(diss_ops)): # copy the dissipation operators, their hermitian conjugates, the coupling constants, and the "constant lindblad term" (described below) to the GPU
            
            self.dissipation_ops.append(gpuarray.to_gpu(np.asarray(diss_ops[ii]).astype(np.complex_)))
            self.dissipation_ops_herm.append( linalg.hermitian(self.dissipation_ops[ii]))
            
            self.const_lindblad_term.append(linalg.dot(self.dissipation_ops_herm[ii], self.dissipation_ops[ii])) # this is the term in the Lindblad Master Equation that does not depend on rho; therefore, we need only compute it when the code is first called.
            
            self.couple_consts.append(gpuarray.to_gpu(np.asarray(coupling_consts[ii]).astype(np.complex_)))
        
        # everything for expected values
        self.e_ops_gpu = []
        self.elt_num_gpu = []
        self.exp_vals_gpu = []
        
        for ii in range(0,len(e_ops)): # copy the evaluation operators to the GPU and create an array, exp_vals_gpu, to store expectation values
            
            self.exp_vals_gpu.append([])
            self.e_ops_gpu.append(gpuarray.to_gpu(np.asarray(e_ops[ii]).astype(np.complex_)))
            
        self.dummy_array = gpuarray.to_gpu(np.zeros_like(rho_initial_cpu, dtype=np.complex_)) # this is used to store values needed for one computation; for example, it is used to compute the terms inside the summand of the Lindblad Master Equation.
        
        self.tolerance = gpuarray.to_gpu(np.asarray(tol))
        
        self.do_adaptive = adapt
       
    
def runrk4_mesolve(rho_initial, tlist, hamiltonian, diss_ops, e_ops, coupling_consts, do_adaptive = 0, tol = 0.1):

    ## This is the function that must be called by the user.
    
    skcuda.misc.init() 
    
    if(do_adaptive == 1):
        dt = 0.5*(tlist[1] - tlist[0])
        
    if(do_adaptive == 0):
        dt = 0.001
    
    gpu = GPU_mesolve_internals(dt, hamiltonian, rho_initial, e_ops, diss_ops, tlist, tol, do_adaptive, coupling_consts) # create an instance of the GPU_mesolve_internals class, which we will use to evolve rho in time.
    
    exp_vals_calc(gpu) # calculate expectation values for time_list[0] = t0
    
    for ii in range(0,len(tlist)-1):
        
        runrk4_mesolve_core(gpu, tlist[ii], tlist[ii+1]) # evolve the density operator between two times specified by the user
        
        exp_vals_calc(gpu) # calculate expectation values for tlist[ii]
        
    return gpu.exp_vals_gpu

def runrk4_mesolve_core(gpu, start_time, final_time):
    
    ## This function is called by the runrk4_mesolve function.  It evolves a state, rho, at time start_time to the state rho_evolved at final_time
    
    current_time = start_time
    
    while current_time < final_time: # run the following loop until we reach final_time
        
        if gpu.dt_dummy > (final_time - current_time): # if our timestep is larger than the remaining time until final_time, then use final_time - current_time *for this step only*
            
            gpu.dt_full = gpuarray.to_gpu(np.asarray([final_time - current_time], dtype=np.complex_))
            
            rk4_gpu_stepper(gpu, 1)
            
            gpu.dt_full = gpuarray.to_gpu(np.asarray([gpu.dt_dummy], dtype=np.complex_))
            gpu.dt_half = 0.5*gpu.dt_full
            
            current_time = final_time
            
        if current_time < final_time: # if we have not yet reached the final_time, then do another computation.  This if statement ensures that we do not execute this code as well as the code in lines 105-114.
            
            if(gpu.do_adaptive == 1):
                adaptive_stepper(gpu, current_time, final_time)
                
            if(gpu.do_adaptive == 0):
                rk4_gpu_stepper(gpu, 1)
        
            current_time += gpu.dt_dummy
    
    return
    
def rk4_gpu_stepper(gpu, half_or_full):
    
    ## This function executes a single timestep computation; we are using Runge-Kutta 4, thus the equation is:
    ## rho_evolved = rho + (1/6)(k1 + 2k2 + 2k3 + k4)
    
    if(half_or_full == 1): # full
        rho = gpu.rho_evolved
        dt = gpu.dt_full
    
    if(half_or_full == 0): # half
        rho = gpu.rho_evolved_for_adaptive
        dt = gpu.dt_half
    
    lindblad(gpu, rho) # compute the terms in the summand of the Lindblad Master Equation, it is stored inside gpu.dummy_array.
    
    k1_gpu = skcuda.misc.multiply(dt, (von_Neumann(gpu.hamiltonian, rho) + gpu.dummy_array))
    
    lindblad(gpu, rho + 0.5*k1_gpu)
    
    k2_gpu = skcuda.misc.multiply(dt, (von_Neumann(gpu.hamiltonian, rho + 0.5*k1_gpu) + gpu.dummy_array))
    
    lindblad(gpu, rho + 0.5*k2_gpu)
    
    k3_gpu = skcuda.misc.multiply(dt, (von_Neumann(gpu.hamiltonian, rho + 0.5*k2_gpu) + gpu.dummy_array))
    
    lindblad(gpu, rho + k3_gpu)
    
    k4_gpu = skcuda.misc.multiply(dt, (von_Neumann(gpu.hamiltonian, rho + k3_gpu) + gpu.dummy_array))
    
    gpu.rho_evolved = gpu.rho_evolved + k1_gpu/6.0 + k2_gpu/3.0 + k3_gpu/3.0 + k4_gpu/6.0
    
    return

def von_Neumann(first_term, second_term):
    
    ## This computes the von Neumann equation (which is the first term in the Lindblad Master Equation). 
    
    return -1j*(linalg.dot(first_term, second_term) - linalg.dot(second_term, first_term))

def lindblad(gpu, density):
    
    ## This computes the Lindblad terms in the Lindblad Master Equation.
    
    gpu.dummy_array.fill(0.0)
    
    for ii in range(0, len(gpu.dissipation_ops)):
        
        a = linalg.dot(density, gpu.dissipation_ops_herm[ii])

        gpu.dummy_array += skcuda.misc.multiply(gpu.couple_consts[ii], linalg.dot(gpu.dissipation_ops[ii], a) - 0.5*(linalg.dot(gpu.const_lindblad_term[ii], density) + linalg.dot(density, gpu.const_lindblad_term[ii])))
        
    return

def exp_vals_calc(gpu):
    
    ## compute expectation values
    
    for ii in range(0, len(gpu.e_ops_gpu)):
        
        gpu.exp_vals_gpu[ii].append(linalg.trace(linalg.dot(gpu.rho_evolved, gpu.e_ops_gpu[ii])))
        
    return

def adaptive_stepper(gpu, current_time, final_time):
    
    ## do an adaptive timestep
    
    gpu.rho_storage = gpu.rho_evolved
    
    rk4_gpu_stepper(gpu, 1) # full_step

    rk4_gpu_stepper(gpu, 0) # half_step
    rk4_gpu_stepper(gpu, 0) # half_step

    difference = gpu.rho_evolved - gpu.rho_evolved_for_adaptive

    error = gpuarray.max(difference.__abs__())
    
    gpu.dt_dummy = min(final_time - current_time, (gpu.dt_full * (gpu.tolerance / error)).get().astype(np.float64))
    
    gpu.dt_full = gpuarray.to_gpu(np.asarray(gpu.dt_dummy, dtype=np.complex_))
    gpu.dt_half = 0.5 * gpu.dt_full
    
    gpu.rho_evolved = gpu.rho_storage
    gpu.rho_evolved_for_adaptive = gpu.rho_storage
    
    rk4_gpu_stepper(gpu, 1)
    
    return