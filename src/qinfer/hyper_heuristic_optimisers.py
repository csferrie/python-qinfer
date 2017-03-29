

## FEATURES ###################################################################

from __future__ import absolute_import
from __future__ import division

## IMPORTS ####################################################################

import numpy as np
import random
from functools import partial
from qinfer.perf_testing import perf_test_multiple
from qinfer import distributions
    
## CLASSES ####################################################################

__all__ = [
    'ParticleSwarmOptimiser'
]

class HyperHeuristicOptimiser(object):
    '''
        A generic hyper-heuristic optimiser class that is inherited by the other optimisation functions.

        :param np.ndarray param_names: The list of parameters that are being searched over.
        :param function fitness_function: The function that is being optimised over, defaults to perf test multiple
        :param function boundary_map: Function to constrain points within some boundary regime
        :param dict funct_args: Arguments to pass to the fitness function
        :param dict funct_kwargs: Keyword arguments to pass to the fitness function
    '''

    def __init__(
                self,
                param_names,
                fitness_function = None,
                boundary_map=None,
                *funct_args,
                **funct_kwargs
                ):
        self._param_names = param_names
        self._n_free_params = len(param_names)
        self._boundary_map = boundary_map
        self._funct_args = funct_args
        self._funct_kwargs = funct_kwargs
        
        if fitness_function is None: # Default to calling perf test multiple
            self._optimisable = PerfTestMultipleAbstractor(
                self._param_names,
                *self._funct_args,
                **self._funct_kwargs
            )
        else: 
            self._fitness_function = partial(fitness_function, *self._funct_args, **self._funct_kwargs)

    # Member function needed for parralelisation
    def fitness_function(self, params):
        return self._fitness_function(params)
    
    def parrallel(self):
        raise NotImplementedError("This optimiser does not have parrallel support. To resolve this issue, level an appropriate criticism at the developer.")

class ParticleSwarmOptimiser(HyperHeuristicOptimiser):
    '''
        A particle swarm optimisation based hyperheuristic
        :param integer n_pso_iterations:
        :param integer n_pso_particles:
        :param 
        :param
    '''
    
    def __call__(self,
        n_pso_iterations=50,
        n_pso_particles=60,
        initial_position_distribution=None,
        initial_velocity_distribution=None,
        omega_v=0.35, 
        phi_p=0.25, 
        phi_g=0.5,
        serial_map=map
        ):
        self._fitness_dt = np.dtype([
            ('params', np.float64, (self._n_free_params,)),
            ('velocities', np.float64, (self._n_free_params,)),
            ('fitness', np.float64)])
        self._fitness = np.empty([n_pso_iterations, n_pso_particles], dtype=self._fitness_dt)
        local_attractors = np.empty([n_pso_particles], dtype=self._fitness_dt)
        global_attractor = np.empty([1], dtype=self._fitness_dt)

        if initial_position_distribution is None:
            initial_position_distribution = distributions.UniformDistribution(np.array([[ 0, 1]] * self._n_free_params));
            
        if initial_velocity_distribution is None:
            initial_velocity_distribution = distributions.UniformDistribution(np.array([[-1, 1]] * self._n_free_params))
        
        # Initial particle positions
        self._fitness[0]["params"] = initial_position_distribution.sample(n_pso_particles)
            
        # Apply the boundary conditions if any exist
        if self._boundary_map is not None:
            self._fitness[itr]["params"] = self._boundary_map(self._fitness[itr]["params"])

        # Calculate the initial particle fitnesses
        self._fitness[0]["fitness"] = self.evaluate_fitness(self._fitness[0]["params"], 
                                                            serial_map=serial_map)

        # Calculate the positions of the attractors
        local_attractors = self._fitness[0]
        local_attractors, global_attractor = self.update_attractors(
                                                self._fitness[0], 
                                                local_attractors, 
                                                global_attractor)

        # Initial particle velocities
        self._fitness[0]["velocities"] = initial_velocity_distribution.sample(n_pso_particles)
        self._fitness[0]["velocities"] = self.update_velocities(
                                                self._fitness[0]["params"], 
                                                self._fitness[0]["velocities"], 
                                                local_attractors["params"],
                                                global_attractor["params"],
                                                omega_v, phi_p, phi_g)

        for itr in range(1, n_pso_iterations):
            #Update the particle positions
            self._fitness[itr]["params"] = self.update_positions(
                self._fitness[itr - 1]["params"], 
                self._fitness[itr - 1]["velocities"])

            # Apply the boundary conditions if any exist
            if self._boundary_map is not None:
                self._fitness[itr]["params"] = self._boundary_map(self._fitness[itr]["params"])

            # Recalculate the fitness function
            self._fitness[itr]["fitness"] = self.evaluate_fitness(
                self._fitness[itr]["params"],
                serial_map=serial_map)

            # Find the new attractors
            local_attractors, global_attractor = self.update_attractors(
                self._fitness[itr], 
                local_attractors, 
                global_attractor)

            # Update the velocities
            self._fitness[itr]["velocities"] = self.update_velocities(
                self._fitness[itr]["params"], 
                self._fitness[itr - 1]["velocities"], 
                local_attractors["params"],
                global_attractor["params"],
                omega_v, phi_p, phi_g)

        return global_attractor

    def evaluate_fitness(self, particles, serial_map):
        fitness_function = partial(self.fitness_function)
        fitness = np.empty([len(particles)], dtype=np.float64)
        fitness = serial_map(self.fitness_function, particles)
        return fitness
        
    def update_positions(self, positions, velocities):
        updated = positions + velocities
        return updated

    def update_velocities(self, positions, velocities, local_attractors, global_attractor, omega_v, phi_p, phi_g):
        random_p = np.random.random_sample(positions.shape)
        random_g = np.random.random_sample(positions.shape)
        updated = omega_v * velocities + phi_p * random_p * (local_attractors - positions) + phi_g * random_g * (global_attractor - positions) 
        return updated

    def update_attractors(self, particles, local_attractors, global_attractor):
        for idx, particle in enumerate(particles):
            if particle["fitness"] < local_attractors[idx]["fitness"]:
                local_attractors[idx] = particle
        global_attractor = local_attractors[np.argmin(local_attractors["fitness"])]
        return local_attractors, global_attractor
    
class PerfTestMultipleAbstractor:
    def __init__(self, 
                 param_names,
                 evaluation_function = None, 
                 *args, 
                 **kwargs):
        self._heuristic = kwargs['heuristic_class']
        del kwargs['heuristic_class']
        self._args = args
        self._kwargs = kwargs
        self._param_names = param_names
        if evaluation_function is None:
            self._evaluation_function = lambda performance: performance['loss'][:,-1].mean(axis=0)
        else:
            self._evaluation_function = evaluation_function
        
    def __call__(self, params):
        performance = perf_test_multiple(
            *self._args,
            heuristic_class = self._heuristic(**{
                name: param
                for name, param in zip(self._param_names, params)
            }),
            **self._kwargs
        )
        return self._evaluation_function(performance)