# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Tests for the levels for the MLMC-MLCV algorithm."""

from __future__ import annotations

from typing import Callable

import pytest
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import array
from numpy import ndarray
from numpy.testing import assert_equal

from gemseo_umdo.statistics.multilevel.mlmc_mlcv.level import Level


@pytest.fixture(scope="module")
def surrogate_model() -> tuple[Callable[[ndarray], ndarray], float]:
    """The surrogate model g."""

    def g(x: ndarray) -> ndarray:
        r"""A surrogate :math:`g_{\ell}` model of the model :math:`f_{\ell}`.

        Args:
            x: The input data.

        Returns:
            The output data.
        """
        return 1.8 * x

    return g, 0.0


@pytest.fixture(scope="module")
def difference_surrogate_model() -> tuple[Callable[[ndarray], ndarray], float]:
    """The surrogate model h."""

    def h(x: ndarray) -> ndarray:
        r"""A surrogate :math:`h_{\ell}` model of :math:`f_{\ell}-f_{\ell-1}`.

        Args:
            x: The input data.

        Returns:
            The output data.
        """
        return 0.05 * x

    return h, 0.0


def test_default(model, surrogate_model, difference_surrogate_model):
    """Check default configuration."""
    level = Level(model, surrogate_model, difference_surrogate_model)

    assert isinstance(level.surrogate_model, tuple)
    assert len(level.surrogate_model) == 2
    g = level.surrogate_model[0]
    assert isinstance(g, MDOFunction)
    assert g.name == "g"
    assert g.evaluate(array([2.0])) == array([3.6])
    assert level.surrogate_model[1] == 0.0

    assert isinstance(level.difference_surrogate_model, tuple)
    assert len(level.difference_surrogate_model) == 2
    h = level.difference_surrogate_model[0]
    assert isinstance(h, MDOFunction)
    assert h.name == "h"
    assert h.evaluate(array([2.0])) == array([0.1])
    assert level.difference_surrogate_model[1] == 0.0

    assert isinstance(level.model, MDOFunction)
    assert level.model.name == "f"
    assert level.model.evaluate(array([1.0])) == array([2.0])

    assert level.cost is None
    assert level.n_cost_estimation_samples == 1
    assert level.n_initial_samples == 10
    assert_equal(level.sampling_ratio, array([2.0, 2.0]))


def test_sampling_ratio(model, surrogate_model, difference_surrogate_model):
    """Check sampling_ratio."""
    level = Level(
        model, surrogate_model, difference_surrogate_model, sampling_ratio=1.3
    )
    assert level.sampling_ratio == 1.3


def test_n_cost_estimation_samples(model, surrogate_model, difference_surrogate_model):
    """Check n_cost_estimation_samples."""
    level = Level(
        model, surrogate_model, difference_surrogate_model, n_cost_estimation_samples=3
    )
    assert level.n_cost_estimation_samples == 3


def test_cost(model, surrogate_model, difference_surrogate_model):
    """Check cost."""
    level = Level(model, surrogate_model, difference_surrogate_model, cost=3)
    assert level.cost == 3


def test_n_initial_samples(model, surrogate_model, difference_surrogate_model):
    """Check n_initial_samples."""
    level = Level(
        model, surrogate_model, difference_surrogate_model, n_initial_samples=3
    )
    assert level.n_initial_samples == 3
