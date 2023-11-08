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
"""A Monte Carlo sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from gemseo.algos.doe.lib_openturns import OpenTURNS
from numpy import array
from numpy import hstack
from numpy import vstack
from numpy.typing import NDArray

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace

FunctionType = Callable[[NDArray[float]], NDArray[float]]


class MonteCarloSampler:
    """A Monte Carlo sampler taking advantage of the vectorized functions."""

    __algo: OpenTURNS
    """The Monte Carlo algorithm."""

    __functions: list[FunctionType]
    """The functions to sample."""

    __input_space: DesignSpace
    """The input space on which to sample the functions."""

    __input_histories: list[NDArray[float]]
    """One history of the function inputs per call to the sampler."""

    __output_histories: list[NDArray[float]]
    """One history of the function outputs per call to the sampler."""

    def __init__(self, input_space: DesignSpace) -> None:
        """
        Args:
            input_space: The input space on which to sample the functions.
        """  # noqa:D205 D212 D415
        self.__algo = OpenTURNS()
        self.__algo.algo_name = "OT_MONTE_CARLO"
        self.__functions = []
        self.__input_space = input_space
        self.__input_histories = []
        self.__output_histories = []
        self.__all_functions_are_vectorized = True

    def add_function(self, function: FunctionType, is_vectorized: bool = True) -> None:
        """Add a function to sample.

        Args:
            function: A function to sample.
            is_vectorized: Whether the function is vectorized.
        """
        self.__all_functions_are_vectorized &= is_vectorized
        self.__functions.append(function)

    def __call__(
        self, n_samples: int, seed: int | None = None
    ) -> tuple[NDArray[float], NDArray[float]]:
        """Sample the functions with a Monte Carlo algorithm.

        Args:
            n_samples: The number of samples.
            seed: The seed value.
                If ``None``,
                use the [OpenTURNS.seed][gemseo.algos.doe.lib_openturns.OpenTURNS.seed].

        Returns:
            The input and output samples.
        """
        input_samples = self.__algo.compute_doe(
            self.__input_space, size=n_samples, seed=seed
        )
        if self.__all_functions_are_vectorized:
            output_samples = [function(input_samples) for function in self.__functions]
        else:
            output_samples = [
                vstack([function(input_sample) for input_sample in input_samples])
                for function in self.__functions
            ]

        output_samples = hstack(output_samples)
        self.__input_histories.append(input_samples)
        self.__output_histories.append(output_samples)
        return input_samples, output_samples

    @property
    def input_history(self) -> NDArray[float]:
        """The history of the function inputs."""
        if self.__input_histories:
            return vstack(self.__input_histories)

        return array(())

    @property
    def output_history(self) -> NDArray[float]:
        """The history of the function outputs."""
        if self.__output_histories:
            return vstack(self.__output_histories)

        return array(())
