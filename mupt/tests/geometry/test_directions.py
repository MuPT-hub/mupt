"""Tests for random coordinate direction utilities."""

import numpy as np

from mupt.geometry.coordinates.directions import random_orthogonal_vector, random_unit_vector, random_vector


def test_random_vector_accepts_repeatable_rng():
    first = random_vector(rng=np.random.default_rng(1234))
    second = random_vector(rng=np.random.default_rng(1234))

    np.testing.assert_allclose(first, second)


def test_random_vector_rng_does_not_mutate_global_numpy_rng():
    np.random.seed(5678)
    expected = np.random.uniform(low=-1.0, high=1.0, size=3)

    np.random.seed(5678)
    random_vector(rng=np.random.default_rng(1234))
    actual = np.random.uniform(low=-1.0, high=1.0, size=3)

    np.testing.assert_allclose(actual, expected)


def test_random_unit_vector_accepts_rng_and_normalizes():
    vector = random_unit_vector(rng=np.random.default_rng(1234))

    np.testing.assert_allclose(np.linalg.norm(vector), 1.0)


def test_random_orthogonal_vector_accepts_repeatable_rng():
    vector = np.array([1.0, 0.0, 0.0])
    first = random_orthogonal_vector(vector, rng=np.random.default_rng(1234))
    second = random_orthogonal_vector(vector, rng=np.random.default_rng(1234))

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(np.dot(first, vector), 0.0, atol=1e-12)
