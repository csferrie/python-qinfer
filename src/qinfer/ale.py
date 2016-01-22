#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# ale.py: Adaptive likelihood estimation utilities and models.
##
# © 2012 Chris Ferrie (csferrie@gmail.com) and
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

# FEATURES ####################################################################

from __future__ import division

# ALL #########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'binom_est_p',
    'ALEApproximateModel'
]

# IMPORTS #####################################################################

from itertools import count
import warnings

import numpy as np

from scipy.stats.distributions import binom

from qinfer.abstract_model import Model, Simulatable
from qinfer._exceptions import ApproximationWarning

## FUNCTIONS ##################################################################


def binom_est_p(number_of_successes, number_of_trials, hedge=float(0)):
    r"""
    Given a number of successes :math:`n` and a number of trials :math:`N`,
    estimates the binomial distribution parameter :math:`p` using the
    hedged maximum likelihood estimator of [FB12]_.
    
    :param number_of_successes: Number of successes.
    :type number_of_successes: `numpy.ndarray` or `int`
    :param int number_of_trials: Number of trials.
    :param float hedge: Hedging parameter :math:`\beta`.
    :rtype: `float` or `numpy.ndarray`.
    :return: The estimated binomial distribution parameter :math:`p` for each
        value of :math:`n`.
    """
    return (number_of_successes + hedge) / (number_of_trials + 2 * hedge)


def binom_est_error(probability_of_success, number_of_trials, hedge=float(0)):
    r"""
    Given a probability of success :math:`p` and a number of trials :math:`N`,
    estimates the error in estimation of :math:`p` using a hedged estimator

    :param float probability_of_success: Probability of success
    :param int number_of_trials: The number of trials
    :param float hedge: Hedging parameter :math:`\beta`
    :rtype: float or `numpy.ndarray`
    :return: The estimated error in the parameter :math:`p` for each value of
        :math:`p`
    """
    # asymptotic np.sqrt(p * (1 - p) / N)
    return np.sqrt(
        probability_of_success * (1 - probability_of_success) /
        (number_of_trials + 2 * hedge + 1)
    )

## CLASSES ####################################################################


class ALEApproximateModel(Model):
    r"""
    Given a :class:`~qinfer.abstract_model.Simulatable`, estiamtes the
    likelihood of that simulator by using adaptive likelihood estimation (ALE).
    
    :param qinfer.abstract_model.Simulatable simulator: Simulator to estimate
        the likelihood function of.
    :param float error_tol: Allowed error in the estimated likelihood. Note that
        the simulation cost scales as :math:`O(\epsilon^{-2})`, where
        :math:`\epsilon` is the error tolerance.
    :param int min_samp: Minimum number of samples to use in estimating the
        likelihood.
    :param int samp_step: Number of samples by which to increment if the error
        tolerance is not met.
    :param float est_hedge: Amount of hedging to use in reporting the final
        estimate.
    :param float adapt_hedge: Amount of hedging to use in deciding if the error
        tolerance has been met. Increasing this parameter will in general
        cause the algorithm to require more samples.
    """
    
    def __init__(self, simulator,
        error_tol=1e-2, min_samp=10, samp_step=10,
        est_hedge=0.509, adapt_hedge=0.509
    ):
        
        ## INPUT VALIDATION ##
        if not isinstance(simulator, Simulatable):
            raise TypeError("Simulator must be an instance of Simulatable.")

        if error_tol <= 0:
            raise ValueError("Error tolerance must be strictly positive.")
        if error_tol > 1:
            raise ValueError("Error tolerance must be less than 1.")
            
        if min_samp <= 0:
            raise ValueError("Minimum number of samples (min_samp) must be positive.")
        if samp_step <= 0:
            raise ValueError("Sample step (samp_step) must be positive.")
        if est_hedge < 0:
            raise ValueError("Estimator hedging (est_hedge) must be non-negative.")
        if adapt_hedge < 0:
            raise ValueError("Adaptive hedging (adapt_hedge) must be non-negative.")
            
        self._simulator = simulator
        # We had to have the simulator in place before we could call
        # the superclass.
        super(ALEApproximateModel, self).__init__()
        
        self._error_tol = float(error_tol)
        self._min_samp = int(min_samp)
        self._samp_step = int(samp_step)
        self._est_hedge = float(est_hedge)
        self._adapt_hedge = float(adapt_hedge)
        
    ## WRAPPED METHODS AND PROPERTIES ##
    # These methods and properties do nothing but pass along to the
    # consumed Simulatable instance, and so we present them here in a
    # compressed form.
    
    @property
    def n_modelparams(self): return self._simulator.n_modelparams

    @property
    def expparams_dtype(self): return self._simulator.expparams_dtype

    @property
    def is_n_outcomes_constant(self): return self._simulator.is_n_outcomes_constant

    @property
    def sim_count(self): return self._simulator.sim_count

    @property
    def Q(self): return self._simulator.Q
    
    def n_outcomes(self, expparams): return self._simulator.n_outcomes(expparams)

    def are_models_valid(self, modelparams): return self._simulator.are_models_valid(modelparams)

    def simulate_experiment(self, modelparams, expparams, repeat=1):
        return self._simulator.simulate_experiment(modelparams, expparams, repeat)

    def experiment_cost(self, expparams): return self._simulator.experiment_cost(expparams)
    
    ## IMPLEMENTATIONS OF MODEL METHODS ##
    
    def likelihood(self, outcomes, modelparams, expparams):
        # FIXME: at present, will proceed until ALL model experiment pairs
        #        are below error tol.
        #        Should disable one-by-one, but that's tricky.
        super(ALEApproximateModel, self).likelihood(outcomes, modelparams, expparams)
        # We will use the fact we have assumed a two-outcome model to make the
        # problem easier. As such, we will rely on the static method 
        # Model.pr0_to_likelihood_array.
        
        # Start off with min_samp samples.
        n = np.zeros((modelparams.shape[0], expparams.shape[0]))
        for N in count(start=self._min_samp, step=self._samp_step):
            sim_data = self._simulator.simulate_experiment(
                modelparams, expparams, repeat=self._samp_step
            )
            n += np.sum(sim_data, axis=0) # Sum over the outcomes axis to find the
                                          # number of 1s.
            error_est_p1 = binom_est_error(
                binom_est_p(n, N, self._adapt_hedge), N, self._adapt_hedge
            )
            if np.all(error_est_p1 < self._error_tol): break
            
        return Model.pr0_to_likelihood_array(outcomes, 1 - binom_est_p(n, N, self._est_hedge))