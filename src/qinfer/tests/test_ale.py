"""
Contains unit tests for :mod:`qinfer.ale`
"""
__author__ = 'Michal Kononenko'

import mock
import numpy as np
import qinfer.ale as ale
from qinfer.tests.base_test import DerandomizedTestCase
from qinfer.example_models import SimplePrecessionModel


class TestAle(DerandomizedTestCase):

    def setUp(self):
        super(DerandomizedTestCase, self).setUp()


class TestDistributionFunctions(TestAle):
    def setUp(self):
        super(TestDistributionFunctions, self).setUp()
        self.n = 10
        self.N = 100
        self.p = 0.5
        self.hedge = float(0.5)


class TestBinomEstP(TestDistributionFunctions):

    def test_binom_est_p(self):
        expected_result = (self.n + self.hedge) / (self.N + 2 * self.hedge)
        self.assertEqual(
            expected_result, ale.binom_est_p(self.n, self.N, self.hedge)
        )


class TestBinomEstError(TestDistributionFunctions):

    def test_binom_est_error(self):
        expected_result = np.sqrt(
            self.p * (1 - self.p)/(self.N + 2 * self.hedge + 1)
        )
        self.assertEqual(
            expected_result, ale.binom_est_error(self.p, self.N, self.hedge)
        )


class TestALEApproximateModelConstructor(TestAle):

    def setUp(self):
        self.model = SimplePrecessionModel()
        self.error_tol = 1e-3
        self.min_samp = 1
        self.samp_step = 10
        self.est_hedge = 0.214
        self.adapt_hedge = 0.215

    def test_ale_model_constructor(self):
        ale_model = ale.ALEApproximateModel(
            self.model, self.error_tol, self.min_samp, self.samp_step,
            self.est_hedge, self.adapt_hedge)

        self.assertIsInstance(ale_model, ale.ALEApproximateModel)

    def test_constructor_bad_model(self):
        bad_model = 'foo'
        self.assertFalse(isinstance(bad_model, ale.ALEApproximateModel))
        with self.assertRaises(TypeError):
            ale.ALEApproximateModel(
                bad_model, self.error_tol, self.min_samp, self.samp_step,
                self.est_hedge, self.adapt_hedge
            )

    def test_constructor_negative_err_tol(self):
        bad_error_tol = -0.5
        self.assertLessEqual(bad_error_tol, 0)

        with self.assertRaises(ValueError):
            ale.ALEApproximateModel(
                self.model, bad_error_tol, self.min_samp, self.samp_step,
                self.est_hedge, self.adapt_hedge
            )

    def test_constructor_err_tol_too_high(self):
        bad_error_tol = 2
        self.assertGreaterEqual(bad_error_tol, 1)

        with self.assertRaises(ValueError):
            ale.ALEApproximateModel(
                self.model, bad_error_tol, self.min_samp, self.samp_step,
                self.est_hedge, self.adapt_hedge
            )

    def test_constructor_min_samp_too_low(self):
        min_samp = -1
        self.assertLessEqual(min_samp, 0)

        with self.assertRaises(ValueError):
            ale.ALEApproximateModel(
                self.model, self.error_tol, min_samp, self.samp_step,
                self.est_hedge, self.adapt_hedge
            )

    def test_constructor_samp_step_too_low(self):
        samp_step = -1
        self.assertLessEqual(samp_step, 0)

        with self.assertRaises(ValueError):
            ale.ALEApproximateModel(
                self.model, self.error_tol, self.min_samp, samp_step,
                self.est_hedge, self.adapt_hedge
            )

    def test_constructor_est_hedge_too_low(self):
        est_hedge = -1
        self.assertLessEqual(est_hedge, 0)

        with self.assertRaises(ValueError):
            ale.ALEApproximateModel(
                self.model, self.error_tol, self.min_samp, self.samp_step,
                est_hedge, self.adapt_hedge
            )

    def test_constructor_adapt_hedge_too_low(self):
        adapt_hedge = -1
        self.assertLessEqual(adapt_hedge, 0)

        with self.assertRaises(ValueError):
            ale.ALEApproximateModel(
                self.model, self.error_tol, self.min_samp, self.samp_step,
                self.est_hedge, adapt_hedge
            )


class TestALEApproximateModel(TestAle):
    def setUp(self):
        self.model = SimplePrecessionModel()
        self.error_tol = 1e-3
        self.min_samp = 1
        self.samp_step = 10
        self.est_hedge = 0.214
        self.adapt_hedge = 0.215

        self.ale_model = ale.ALEApproximateModel(
            self.model, self.error_tol, self.min_samp,
            self.samp_step, self.est_hedge, self.adapt_hedge)


class TestALEProperties(TestALEApproximateModel):

    def test_n_model_params(self):
        self.assertEqual(
            self.ale_model.n_modelparams,
            self.ale_model._simulator.n_modelparams
        )

    def test_expparams_type(self):
        self.assertEqual(
            self.ale_model.expparams_dtype,
            self.ale_model._simulator.expparams_dtype
        )

    def test_is_n_outcomes_constant(self):
        self.assertEqual(
            self.ale_model.is_n_outcomes_constant,
            self.ale_model._simulator.is_n_outcomes_constant
        )

    def test_sim_count(self):
        self.assertEqual(
            self.ale_model.sim_count,
            self.ale_model._simulator.sim_count
        )

    def test_Q(self):
        self.assertEqual(
            self.ale_model.Q,
            self.ale_model._simulator.Q
        )


class TestNParams(TestALEApproximateModel):

    def setUp(self):
        TestALEApproximateModel.setUp(self)
        self.n_outcomes = 3
        self.expparams = [10, 11, 12]
        self.ale_model._simulator.n_outcomes = mock.MagicMock(
            return_value=self.n_outcomes)

    def test_n_params(self):
        self.assertEqual(self.n_outcomes,
                         self.ale_model.n_outcomes(self.expparams))
        self.assertEqual(mock.call(self.expparams),
                         self.ale_model._simulator.n_outcomes.call_args)


class TestAreModelsValid(TestALEApproximateModel):

    def setUp(self):
        TestALEApproximateModel.setUp(self)
        self.are_models_valid = True
        self.model_params = [10, 11, 12]
        self.ale_model._simulator.are_models_valid = mock.MagicMock(
            return_value=self.are_models_valid
        )

    def test_are_models_valid(self):
        self.assertEqual(self.are_models_valid,
                         self.ale_model.are_models_valid(self.model_params))
        self.assertEqual(mock.call(self.model_params),
                         self.ale_model._simulator.are_models_valid.call_args)


class TestSimulateExperiment(TestALEApproximateModel):
    def setUp(self):
        TestALEApproximateModel.setUp(self)
        self.modelparams = [10, 11, 12]
        self.expparams = [20, 21, 22]
        self.repeat = 10

        self.return_value = 10

        self.ale_model._simulator.simulate_experiment = mock.MagicMock(
            return_value=self.return_value
        )

    def test_simulate_experiment(self):
        self.assertEqual(
            self.ale_model.simulate_experiment(self.modelparams, self.expparams,
                                               repeat=self.repeat),
            self.return_value
        )
        self.assertEqual(
            mock.call(self.modelparams, self.expparams, self.repeat),
            self.ale_model._simulator.simulate_experiment.call_args
        )


class TestExperimentCost(TestALEApproximateModel):
    def setUp(self):
        TestALEApproximateModel.setUp(self)
        self.expparams = [30, 31, 32]

        self.return_value = [40, 41, 42]
        self.ale_model._simulator.experiment_cost = mock.MagicMock(
            return_value=self.return_value)

    def test_experiment_cost(self):
        self.assertEqual(
            self.ale_model.experiment_cost(self.expparams), self.return_value
        )

        self.assertEqual(
            mock.call(self.expparams),
            self.ale_model._simulator.experiment_cost.call_args
        )


class TestLikelihood(TestALEApproximateModel):
    def setUp(self):
        TestALEApproximateModel.setUp(self)

        self.outcomes = np.array([10, 20, 30])
        self.modelparams = np.array([11, 21, 31])
        self.expparams =  np.array([41, 51, 61])

        self.number_of_ones = 5

        self.ale_model._simulator.simulate_experiment = mock.MagicMock(
            return_value=np.ones([1, self.number_of_ones])
        )

        self.ale_model._error_tol = 0.1
