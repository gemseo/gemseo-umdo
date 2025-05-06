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
"""Defining the uncertain space of the truss problem."""

from __future__ import annotations

from math import log
from math import pi
from math import sqrt

from gemseo.algos.parameter_space import ParameterSpace


class TrussUncertainSpace(ParameterSpace):
    """The uncertain space of the truss problem."""

    def __init__(self, factor: float = 1.0, use_different_bars: bool = False) -> None:
        """
        Args:
            factor: The multiplication factor of the standard deviation
                with respect to the original paper.
            use_different_bars: Whether the bars are different.
                Otherwise, use identical oblical bars and identical horizontal bars.
        """  # noqa: D205 D212
        super().__init__()
        for name in (
            (f"A1_{i}" for i in range(1, 12)) if use_different_bars else ("A1",)
        ):
            self.add_random_variable(
                name,
                "OTDistribution",
                interfaced_distribution="LogNormal",
                interfaced_distribution_parameters=self._to_lognormal(
                    2e-3, factor * 2e-4
                ),
            )

        for name in (
            (f"A2_{i}" for i in range(1, 13)) if use_different_bars else ("A2",)
        ):
            self.add_random_variable(
                name,
                "OTDistribution",
                interfaced_distribution="LogNormal",
                interfaced_distribution_parameters=self._to_lognormal(
                    1e-3, factor * 2e-4
                ),
            )

        for name in (
            (
                *(f"E1_{i}" for i in range(1, 12)),
                *(f"E2_{i}" for i in range(1, 13)),
            )
            if use_different_bars
            else ("E1", "E2")
        ):
            self.add_random_variable(
                name,
                "OTDistribution",
                interfaced_distribution="LogNormal",
                interfaced_distribution_parameters=self._to_lognormal(
                    2.1e11, factor * 2.1e10
                ),
            )

        for i in range(1, 7):
            self.add_random_variable(
                f"P{i}",
                "OTDistribution",
                interfaced_distribution="Gumbel",
                interfaced_distribution_parameters=self._to_gumbel(5e4, factor * 7.5e3),
            )

    @staticmethod
    def _to_lognormal(mean, std):
        """Transformation of mean and std for log-normal law."""
        log_mean = log(mean / sqrt(1 + (std / mean) ** 2))
        log_std = sqrt(log(1 + (std / mean) ** 2))
        return log_mean, log_std

    @staticmethod
    def _to_gumbel(mean, std):
        """Transformation of mean and std for gumbel law."""
        beta = std / pi * sqrt(6)
        gamma = mean - 0.57721 * beta
        return beta, gamma
