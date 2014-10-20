#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# mae.py: Model averaging by sequential Monte Carlo.
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
    'MAEUpdater',
]

## IMPORTS ####################################################################

from itertools import starmap
import warnings

import numpy as np

from qinfer.abstract_model import Model
from qinfer.abstract_updater import Updater
from qinfer.distributions import Distribution
from qinfer._exceptions import ApproximationWarning

try:
    import matplotlib.pyplot as plt
except ImportError:
    import warnings
    warnings.warn("Could not import pyplot. Plotting methods will not work.")
    plt = None

## CLASSES ####################################################################

class MAEUpdater(Updater):
    r"""
    Updates a posterior distribution over several models, along with parameters
    of each model, such that estimates of those parameters can be obtained
    by taking an expectation over models.
    
    All models being considered must admit the
    same experiment parameters (see :attr:`~qinfer.abstract_model.Simulatable.expparams_dtype`).
    Model parameters are assumed to share a common interpretation, such that
    an expectation value over models is well-motivated. If the number of
    model parameters is not constant for the list of models being considered,
    then this updater will only consider the parameters admitted by the model
    with the least parameters. Note that this may be dynamic, if models are
    added or removed during the operation of the updater.
    
    Rather than write a single prior distribution over models, the conditional prior
    distributions :math:`\pi(\vec{x} | M_i)` are taken for each model :math:`M_i`.
    The prior distribution over :math:`M_i` is assumed to be uniform, in order
    to enable adding and removing models dynamically. If the prior over
    :math:`\vec{x}` is independent of :math:`M`, then a single prior can be
    passed in, instead of a list.

    :param list models: List of :class:`~qinfer.abstract_model.Model` instances
        representing the models to be averaged over.
    :param int n_particles: The number of particles to be used in the
        particle approximation for each model.
    :param priors: A representation of the prior distribution.
    :type priors: :class:`~qinfer.distributions.Distribution`, or a `list` of
        distributions
    :param dict smc_kwargs: Dictionary of additional keyword arguments for
        the conditional :class:`~qinfer.smc.SMCUpdater` instances for each
        model.
    :param type updater_class: Actual class to use for conditional distributions.
        This will almost always be :class:`~qinfer.smc.SMCUpdater`, but is allowed
        to vary to enable other approximations.
    :param int n_exclude_params: Specifies the number of parameters to exclude
        from being treated as common. Useful if the individual models aren't
        properly "nested."
    """
    
    ## DESIGN NOTES ###########################################################
    #  We want to maintain the advantages of SMC, namely that everything
    #  proceeds iteratively. Thus, we maintain a prior over models and update
    #  it with each incoming call to update(). This also avoids numerical
    #  instability in the estimation of Pr(m_i | D), as that involves a
    #  normalization over very small numbers as |D| → infinity.
    #
    #  We implement this by recording model weights along with the model
    #  updaters, similarly to SMC itself (each model is thus one gigantic
    #  particle).
    #
    #  To allow for dynamic model addition, this then requires that when a new
    #  model is added, its weight is immediately calculated from the data
    #  record. Moreover, we need a data structure that can manage these weights
    #  when models can be added and removed in an online fashion, and so we
    #  use a dictionary 
    #      _posterior : models → float × updaters
    #  to represent everything.
    #
    #  The weights are kept to a normalization not of 1, but of the total number
    #  of models in the posterior, such that a uniform prior assigns a weight of
    #  1 to each model as it is added.
    
    ## INIT ###################################################################
    
    def __init__(self,
            models, n_particles, priors,
            resampler=None, smc_kwargs={},
            updater_class=SMCUpdater,
            n_exclude_params=0
            ):
            
        super(MAEUpdater, self).__init__()
        
        # Make sure we have a list of priors.
        if isinstance(priors, Distribution):
            priors = [priors] * len(models)
        
        # Remember the things we need to make new SMCUpdater instances.
        self._n_particles = n_particles
        self._smc_kwargs = smc_kwargs
        self._updater_class = updater_class
        
        # Remember details about how the models relate to each other.
        self._n_exclude = n_exclude_params
        
        # Save the models as a dictionary onto updaters.
        self._posterior = {}
        starmap(self.add_model, zip(models, priors))
     
    ## MODEL ADDITION AND REMOVAL #############################################
        
    def add_model(self, model, prior):
        if model not in self._posterior:
            # If we already have models, make sure the new one matches in
            # expparams_dtype.
            if self._posterior:
                current_dtype = self._posterior.iterkeys().next().expparams_dtype
                if model.expparams_dtype != current_dtype:
                    raise TypeError((
                            "Model {} has expparams_dtype {}, which "
                            "doesn't match {} of already added model."
                        ).format(model, model.expparams_dtype, current_dtype)
                    )
            
            # Now that we've checked, we're good to go.
            new_updater = self._updater_class(
                model, self._n_particles, prior, **self._smc_kwargs
            )
            # Add the updater to the posterior with weight 1.
            self._posterior[model] = (1.0, new_updater)
        
            # Do some more validity checking.
            if not self.n_common_modelparams > 0:
                raise ValueError("Models admit less than 1 common parameter--- there is nothing to update.")
            
            if self.data_record:
                raise NotImplementedError("Dynamic addition of models has not yet been implemented; need to work out the weights of the new models.")
            
            # Next, we need to update the new updater with all of the
            # already-collected data.
            new_updater.batch_update(self.data_record, self.experiment_record)
            
    def remove_model(self, model):
        raise NotImplementedError("Not yet implemented.")

    ## PROPERTIES #############################################################

    @property
    def log_total_likelihood(self):
        """
        Returns the log-likelihood of all the data collected so far.
        
        :rtype: `float`
        """
        raise NotImplementedError("Not implemented yet.")
        #return np.sum(np.log(self.normalization_record))
        
    @property
    def worst_n_ess(self):
        """
        Estimates the effective sample size (ESS) of the worst of the posterior
        conditional updaters. This is often useful in determining if the SMC
        approximation has failed, or if the resampling has failed for a
        particular conditional posterior.

        :return float: The smallest effective sample size of any current model.
        """
        return min(updater.ess for weight, updater in self._posterior.itervalues())

    @property
    def data_record(self):
        """
        List of outcomes given to :meth:`~SMCUpdater.update`.
        """
        # We use [:] to force a new list to be made, decoupling
        # this property from the caller.
        return self._data_record[:]
        
    @property
    def n_common_modelparams(self):
        return min(model.n_modelparams for model in self._posterior.iterkeys()) - self._n_exclude

    ## UPDATE METHODS #########################################################

    def update(self, outcome, expparams, check_for_resample=True):
        """
        Given an experiment and an outcome of that experiment, updates the
        posterior distribution to reflect knowledge of that experiment.

        After updating, resamples the posterior distribution if necessary.

        :param int outcome: Label for the outcome that was observed, as defined
            by the :class:`~qinfer.abstract_model.Model` instance under study.
        :param expparams: Parameters describing the experiment that was
            performed.
        :type expparams: :class:`~numpy.ndarray` of dtype given by the
            :attr:`~qinfer.abstract_model.Model.expparams_dtype` property
            of the underlying model
        :param bool check_for_resample: If :obj:`True`, after performing the
            update, the effective sample size condition will be checked and
            a resampling step may be performed.
        """
        super(MAEUpdater, self).update(outcome, expparams)
        
        norm_acc = 0.
        
        for weight, updater in self._posterior.itervalues():
            updater.update(outcome, expparams, check_for_resample=check_for_resample)
            norm_acc += updater.normalization_record[-1]
        
        # Choose a scale such that sum(weights) = len(posterior).
        scale = len(self._posterior) / norm_acc
        
        # Rescale the weights by new normalziation.
        for model, (weight, updater) in self._posterior.iteritems():
            updater[model] = (
                weight * updater.normalization_record[-1],
                updater
            )

    def batch_update(self, outcomes, expparams, resample_interval=5):
        r"""
        Updates based on a batch of outcomes and experiments, rather than just
        one.

        :param numpy.ndarray outcomes: An array of outcomes of the experiments that
            were performed.
        :param numpy.ndarray expparams: Either a scalar or record single-index
            array of experiments that were performed.
        :param int resample_interval: Controls how often to check whether
            :math:`N_{\text{ess}}` falls below the resample threshold.
        """
        raise NotImplementedError("Not yet implemented.")

    ## RESAMPLING METHODS #####################################################

    def resample(self):
        raise NotImplementedError("Not yet implemented.")

    ## DISTRIBUTION CONTRACT ##################################################
    
    @property
    def n_rvs(self):
        return self._model.n_modelparams
        
    def sample(self, n=1):
        # TODO!
        # 1) Pick model.
        # 2) Sample that model's updater.
        # 3) Slice down to common parameters.
        
        # TODO: cache this.
        cumsum_weights = np.cumsum(self.particle_weights)
        return self.particle_locations[np.minimum(cumsum_weights.searchsorted(
            np.random.random((n,)),
            side='right'
        ), len(cumsum_weights) - 1)]

    ## ESTIMATION METHODS #####################################################

    def posterior_model_pr(self):
        """
        Returns the posterior over models, assuming a uniform prior.
        
        :rtype: :class:`numpy.ndarray`, shape ``(len(_updaters),)``.
        :returns: An array containing the posterior probabilities of each model.
        """

        # FIXME: there's a subtle issue here, in that dict does not guarantee
        #        an ordering as elements are added/removed, so the order
        #        of the elements here is arbitrary.
        return np.array([
            weight for weight, updater in self._posterior.itervalues()
        ]) / len(self._posterior)
        
    def est_mean(self):
        """
        Returns an estimate of the posterior mean model, given by the
        expectation value over the current SMC approximation of the posterior
        model distribution.
        
        :rtype: :class:`numpy.ndarray`, shape ``(n_modelparams,)``.
        :returns: An array containing the an estimate of the mean model vector.
        """
        means = np.array([
            weight * updater.est_mean()[:self.n_common_modelparams()]
            for weight, updater in self._posterior.itervalues()
        ])        
            
        # Sum over the model axis, leaving the common model parameter axis.
        return np.sum(means, axis=0)
        
    def est_meanfn(self, fn):
        """
        Returns an estimate of the expectation value of a given function
        :math:`f` of the model parameters, given by a sum over the current SMC
        approximation of the posterior distribution over models.
        
        Here, :math:`f` is represented by a function ``fn`` that is vectorized
        over particles, such that ``f(modelparams)`` has shape
        ``(n_particles, k)``, where ``n_particles = modelparams.shape[0]``, and
        where ``k`` is a positive integer.
        
        :param callable fn: Function implementing :math:`f` in a vectorized
            manner. (See above.)
        
        :rtype: :class:`numpy.ndarray`, shape ``(k, )``.
        :returns: An array containing the an estimate of the mean of :math:`f`.
        """
        raise NotImplementedError("Not yet implemented.")

    def est_covariance_mtx(self):
        """
        Returns an estimate of the covariance of the current posterior model
        distribution, given by the covariance of the current SMC approximation.
        
        :rtype: :class:`numpy.ndarray`, shape
            ``(n_modelparams, n_modelparams)``.
        :returns: An array containing the estimated covariance matrix.
        """
        raise NotImplementedError("Not yet implemented.")

    def est_entropy(self):
        raise NotImplementedError("Not yet implemented.")
        
    def est_kl_divergence(self, other, kernel=None, delta=1e-2):
        raise NotImplementedError("Not yet implemented.")
    
    ## REGION ESTIMATION METHODS ##############################################

    def est_credible_region(self, level=0.95):
        """
        Returns an array containing particles inside a credible region of a
        given level, such that the described region has probability mass
        no less than the desired level.
        
        Particles in the returned region are selected by including the highest-
        weight particles first until the desired credibility level is reached.
        
        :rtype: :class:`numpy.ndarray`, shape ``(n_credible, n_modelparams)``,
            where ``n_credible`` is the number of particles in the credible
            region
        :returns: An array of particles inside the estimated credible region.
        """
        
        raise NotImplementedError("Not yet implemented.")
    
    def region_est_hull(self, level=0.95):
        """
        Estimates a credible region over models by taking the convex hull of
        a credible subset of particles.
        
        :param float level: The desired crediblity level (see
            :meth:`SMCUpdater.est_credible_region`).
        """
        raise NotImplementedError("Not yet implemented.")

    def region_est_ellipsoid(self, level=0.95, tol=0.0001):
        """
        Estimates a credible region over models by finding the minimum volume
        enclosing ellipse (MVEE) of a credible subset of particles.
        
        
        :param float level: The desired crediblity level (see
            :meth:`SMCUpdater.est_credible_region`).
        :param float tol: The allowed error tolerance in the MVEE optimization
            (see :meth:`~qinfer.utils.mvee`).
        """
        raise NotImplementedError("Not yet implemented.")
        
    ## PLOTTING METHODS #######################################################
    
    def posterior_mesh(self, idx_param1=0, idx_param2=1, res1=100, res2=100, smoothing=0.01):
        """
        Returns a mesh, useful for plotting, of kernel density estimation
        of a 2D projection of the current posterior distribution.
        
        :param int idx_param1: Parameter to be treated as :math:`x` when
            plotting.
        :param int idx_param2: Parameter to be treated as :math:`y` when
            plotting.
        :param int res1: Resolution along the :math:`x` direction.
        :param int res2: Resolution along the :math:`y` direction.
        :param float smoothing: Standard deviation of the Gaussian kernel
            used to smooth the particle approximation to the current posterior.
            
        .. seealso::
        
            :meth:`SMCUpdater.plot_posterior_contour`
        """
        
        raise NotImplementedError("Not yet implemented.")
    
    def plot_posterior_contour(self, idx_param1=0, idx_param2=1, res1=100, res2=100, smoothing=0.01):
        """
        Plots a contour of the kernel density estimation
        of a 2D projection of the current posterior distribution.
        
        :param int idx_param1: Parameter to be treated as :math:`x` when
            plotting.
        :param int idx_param2: Parameter to be treated as :math:`y` when
            plotting.
        :param int res1: Resolution along the :math:`x` direction.
        :param int res2: Resolution along the :math:`y` direction.
        :param float smoothing: Standard deviation of the Gaussian kernel
            used to smooth the particle approximation to the current posterior.
            
        .. seealso::
        
            :meth:`SMCUpdater.posterior_mesh`
        """
        raise NotImplementedError("Not yet implemented.")
        
    ## IPYTHON SUPPORT METHODS ################################################
    
    def _repr_html_(self):
        raise NotImplementedError("Not yet implemented.")
        
        return r"""
        <strong>{cls_name} for model of type {model}:</strong>
        <table>
            <caption>Current estimated parameters</caption>
            <thead>
                <tr>
                    {parameter_names}
                </tr>
            </thead>
            <tbody>
                <tr>
                    {parameter_values}
                </tr>
            </tbody>
        </table>
        <em>Resample count:</em> {resample_count}
        """.format(
            cls_name=type(self).__name__, # Useful for subclassing.
            model=type(self.model).__name__,
            
            parameter_names="\n".join(
                map("<td>${}$</td>".format, self.model.modelparam_names)
            ),
            
            # TODO: change format string based on number of digits of precision
            #       admitted by the variance.
            parameter_values="\n".join(
                "<td>{}</td>".format(
                    format_uncertainty(mu, std)
                )
                for mu, std in
                zip(self.est_mean(), np.sqrt(np.diag(self.est_covariance_mtx())))
            ),
            
            resample_count=self.resample_count
        )
        
