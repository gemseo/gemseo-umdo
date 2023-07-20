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
"""Tests for the levels for the MLMC algorithm."""
from __future__ import annotations

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo_umdo.statistics.multilevel.mlmc.level import Level
from numpy import array


def test_default(model):
    """Check default configuration."""
    level = Level(model)
    assert level.model(array([1.0])) == array([2.0])
    assert isinstance(level.model, MDOFunction)
    assert level.model.name == "f"
    assert level.cost is None
    assert level.n_cost_estimation_samples == 1
    assert level.n_initial_samples == 10
    assert level.sampling_ratio == 2.0


def test_sampling_ratio(model):
    """Check sampling_ratio."""
    level = Level(model, sampling_ratio=1.3)
    assert level.sampling_ratio == 1.3


def test_n_cost_estimation_samples(model):
    """Check n_cost_estimation_samples."""
    level = Level(model, n_cost_estimation_samples=3)
    assert level.n_cost_estimation_samples == 3


def test_cost(model):
    """Check custom cost."""
    level = Level(model, cost=3)
    assert level.cost == 3


def test_n_initial_samples(model):
    """Check n_initial_samples."""
    level = Level(model, n_initial_samples=3)
    assert level.n_initial_samples == 3
