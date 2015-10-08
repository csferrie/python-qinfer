"""
Contains unit tests for :mod:`qinfer.clustering`
"""
import unittest
import qinfer.clustering as cluster
__author__ = 'Michal Kononenko'


class TestParticleClusters(unittest.TestCase):
    def setUp(self):
        self.particle_locations = range(0, 1000)

    def test_value_error(self):
        """
        Tests that :exc:`ValueError` is thrown if the ``weighted`` flag
        in :function:`qinfer.clustering.particle_clusters` is ``True``,
        and no particle weights are provided
        """
        with self.assertRaises(ValueError):
            for _ in cluster.particle_clusters(
                    self.particle_locations, weighted=True):
                pass

    def test_unweighted_distribution(self):
        for idx_clstr, clstr in cluster.particle_clusters(
                self.particle_locations, quiet=False):
            print(idx_clstr)
