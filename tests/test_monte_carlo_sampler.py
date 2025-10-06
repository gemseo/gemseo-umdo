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
"""Test the Monte Carlo sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gemseo.algos.design_space import DesignSpace
from numpy import array
from numpy import array_equal
from numpy import newaxis
from numpy.testing import assert_equal

from gemseo_umdo.monte_carlo_sampler import MonteCarloSampler

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo_umdo.monte_carlo_sampler import FunctionType


@pytest.fixture
def input_space() -> DesignSpace:
    """The input space on which to sample the functions."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=2, lower_bound=0.0, upper_bound=1.0)
    return design_space


@pytest.fixture
def functions() -> tuple[FunctionType, FunctionType]:
    """The functions to be sampled."""

    def f(x: NDArray[float]) -> NDArray[float]:
        return x

    def g(x: NDArray[float]) -> NDArray[float]:
        if x.ndim == 2:
            return x.sum(1)[:, newaxis]

        return array([x.sum(0)])

    f.evaluate = f
    g.evaluate = g

    return f, g


@pytest.fixture
def sampler(
    input_space: DesignSpace, functions: list[FunctionType]
) -> MonteCarloSampler:
    """A Monte Carlo sampler."""
    sampler = MonteCarloSampler(input_space)
    sampler.add_function(functions[0])
    sampler.add_function(functions[1])
    return sampler


def test_before_call(sampler):
    """Check the MonteCarloSampler before any call."""
    assert sampler.input_history.size == 0
    assert sampler.output_history.size == 0


def test_add_function(input_space):
    """Check the method add_function."""
    sampler = MonteCarloSampler(input_space)
    assert sampler._MonteCarloSampler__all_functions_are_vectorized
    # All functions are assumed to be vectorized.

    sampler.add_function(lambda x: x)
    assert sampler._MonteCarloSampler__all_functions_are_vectorized
    # All functions are vectorized.

    sampler.add_function(lambda x: x, False)
    assert not sampler._MonteCarloSampler__all_functions_are_vectorized
    # A function is not vectorized.

    sampler.add_function(lambda x: x)
    assert not sampler._MonteCarloSampler__all_functions_are_vectorized
    # A function is not vectorized.


def test_call(sampler):
    """Check __call__."""
    input_samples, output_samples = sampler(3)
    assert input_samples.shape == (3, 2)
    assert output_samples.shape == (3, 3)


def test_histories(sampler):
    """Check input_history and output_history."""
    input_samples, output_samples = sampler(3)
    new_input_samples, new_output_samples = sampler(3)
    assert sampler.input_history.shape == (6, 2)
    assert sampler.output_history.shape == (6, 3)

    assert_equal(sampler.input_history[:3], input_samples)
    assert_equal(sampler.input_history[3:], new_input_samples)
    assert not array_equal(sampler.input_history[3:], sampler.input_history[:3])

    assert_equal(sampler.output_history[:3], output_samples)
    assert_equal(sampler.output_history[3:], new_output_samples)
    assert not array_equal(sampler.output_history[3:], sampler.output_history[:3])


def test_call_seed(sampler):
    """Check __call__ with a new seed."""
    input_samples, output_samples = sampler(3)
    new_input_samples, new_output_samples = sampler(3)
    assert not array_equal(input_samples, new_input_samples)
    assert not array_equal(output_samples, new_output_samples)
    # The seed is incremented at each call.

    new_input_samples, new_output_samples = sampler(3, seed=1)
    assert array_equal(input_samples, new_input_samples)
    assert array_equal(output_samples, new_output_samples)
    # The default seed is 0 and at the first call, it is incremented to 1.


def test_call_vectorized(input_space, sampler, functions):
    """Check __call__ with a non vectorized function."""
    input_samples, output_samples = sampler(3)
    new_sampler = MonteCarloSampler(input_space)
    new_sampler.add_function(functions[0], is_vectorized=False)
    new_sampler.add_function(functions[1])
    new_input_samples, new_output_samples = new_sampler(3)
    assert_equal(input_samples, new_input_samples)
    assert_equal(output_samples, new_output_samples)
