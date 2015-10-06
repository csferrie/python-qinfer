"""
Contains unit tests for :mod:`qinfer.ale`
"""
__author__ = 'Michal Kononenko'

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


class TestALEApproximateModel(TestALEApproximateModelConstructor):
    def setUp(self):
        TestALEApproximateModelConstructor.setUp(self)
        self.ale_model = ale.ALEApproximateModel(
            self.model, self.error_tol, self.min_samp,
            self.samp_step, self.est_hedge, self.adapt_hedge)

