#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_distributions.py: Checks that distribution objects act as expected.
##
# © 2014 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@gmail.com)
#
# This file is a part of the Qinfer project.
# Licensed under the AGPL version 3.
##
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

## FEATURES ###################################################################

from __future__ import absolute_import
from __future__ import division # Ensures that a/b is always a float.

## IMPORTS ####################################################################

import numpy as np

from qinfer.tests.base_test import DerandomizedTestCase

import qinfer.rb as rb
import qinfer.distributions as dist

from qinfer.hyper_heuristic_optimisers import (
    ParticleSwarmOptimizer,
    ParticleSwarmSimpleAnnealingOptimizer,
    ParticleSwarmTemperingOptimizer
)
from qinfer.expdesign import ExpSparseHeuristic

## CLASSES ####################################################################

class OptimizerTestMethods(object):
    # See http://stackoverflow.com/a/1323554/267841 for why this works.

    optimizer_class = None

    def test_quad(self):
        f_quad = lambda x: np.sum(10 * (x - 0.5) ** 2)
        hh_opt = self.optimizer_class(['x', 'y', 'z', 'a'], fitness_function=f_quad)
        hh_opt()

    def test_sin_sq(self):
        f_sin_sq = lambda x: np.sum(np.sin(x - 0.2) ** 2)
        hh_opt = self.optimizer_class(['x', 'y', 'z', 'a'], fitness_function=f_sin_sq)
        hh_opt()

    def test_rosenbrock(self):
        f_rosenbrock = lambda x: np.sum([
            ((x[i + 1]  - x[i] ** 2) ** 2 + (1 - x[i])** 2) / len(x)
            for i in range(len(x) - 1)
        ])
        hh_opt = self.optimizer_class(['x', 'y', 'z', 'a'], fitness_function=f_rosenbrock)
        hh_opt()


    def test_perf_test_multiple_short(self):
        # Define our experiment
        n_trials = 20 # Times we repeat the set of experiments
        n_exp = 100 # Number of experiments in the set
        n_particles = 4000 # Number of points we track during the experiment

        # Model for the experiment
        model = rb.RandomizedBenchmarkingModel()

        #Ordering of RB is 'p', 'A', 'B'
        # A + B < 1, 0 < p < 1
        #Prior distribution of the experiment
        prior = dist.PostselectedDistribution(
            dist.MultivariateNormalDistribution(mean=[0.5,0.1,0.25], cov=np.diag([0.1, 0.1, 0.1])),
            model
        )

        #Heuristic used in the experiment
        heuristic_class = ExpSparseHeuristic

        #Heuristic Parameters
        params = ['base', 'scale']

        #Fitness function to evaluate the performance of the experiment
        hh_opt = self.optimizer_class(params,
                                      n_trials=n_trials,
                                      n_particles=n_particles,
                                      prior=prior,
                                      model=model,
                                      n_exp=n_exp,
                                      heuristic_class=heuristic_class
                                     )
        hh_opt(n_pso_iterations=5,
               n_pso_particles=6)

class TestPSO(DerandomizedTestCase, OptimizerTestMethods):
    optimizer_class = ParticleSwarmOptimizer

class TestPSSAO(DerandomizedTestCase, OptimizerTestMethods):
    optimizer_class = ParticleSwarmSimpleAnnealingOptimizer

class TestPSTO(DerandomizedTestCase, OptimizerTestMethods):
    optimizer_class = ParticleSwarmTemperingOptimizer
