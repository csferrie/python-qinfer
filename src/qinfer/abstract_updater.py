#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# abstract_updater.py: Base class for SMCUpdater, MAEUpdater, etc.
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

## FEATURES ###################################################################

from __future__ import division

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'Updater',
]

## IMPORTS ####################################################################

import numpy as np

from qinfer.distributions import Distribution

## CLASSES #####################################################################

class Updater(Distribution):
    r"""
    Base class for posterior distribution updaters.
    """
    
    def __init__(self):
        self._data_record = []
        self._exp_record  = []
    
    ## DESIGN NOTES ###########################################################
    # This class leaves as abstract the Distribution contract, such that
    # concrete subclasses must implement n_rvs and sample.
    
    ## TODO: ##################################################################
    # - Add abstract methods for any additional things an Updater has to
    #   define.
    # - Move batch_update logic into the base class out of SMCUpdater.
    
    ## SPECIAL METHODS ########################################################
    
    def __len__(self):
        return len(self.data_record)
    
    ## PROPERTIES #############################################################

    @property
    def data_record(self):
        """
        Returns the data that has been used to update this posterior
        distribution.
        
        :rtype: `np.ndarray` of outcomes
        """
        return np.array(self._data_record)

    @property
    def experiment_record(self):
        """
        Returns the experiments that have been used to update this posterior
        distribution.
        
        :rtype: `np.ndarray` of experiments
        """
        return np.array(self._exp_record)
    
    ## UPDATE METHODS #########################################################

    def update(self, outcome, expparams):
        """
        Given an experiment and an outcome of that experiment, updates the
        posterior distribution to reflect knowledge of that experiment.

        :param int outcome: Label for the outcome that was observed, as defined
            by the :class:`~qinfer.abstract_model.Model` instance under study.
        :param expparams: Parameters describing the experiment that was
            performed.
        :type expparams: :class:`~numpy.ndarray` of dtype given by the
            :attr:`~qinfer.abstract_model.Model.expparams_dtype` property
            of the underlying model
        """

        # First, record the outcome.
        self._data_record.append(outcome)
        self._exp_record.append(expparams)

