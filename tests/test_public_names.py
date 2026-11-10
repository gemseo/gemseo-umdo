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
"""Tests for the names exported by the public packages."""

from __future__ import annotations

import pytest

import gemseo_umdo.formulation
import gemseo_umdo.scenario
import gemseo_umdo.statistic


@pytest.mark.parametrize(
    ("package", "names"),
    [
        (
            gemseo_umdo.formulation,
            {
                "ControlVariate_Settings",
                "DOEAlgorithmName",
                "PCE_Settings",
                "Sampling_Settings",
                "SequentialSampling_Settings",
                "Surrogate_Settings",
                "TaylorPolynomial_Settings",
            },
        ),
        (gemseo_umdo.scenario, {"UEvaluationScenario", "UMDOScenario"}),
        (
            gemseo_umdo.statistic,
            {
                "Margin_Settings",
                "Mean_Settings",
                "Probability_Settings",
                "Quantile_Settings",
                "StandardDeviation_Settings",
                "Variance_Settings",
            },
        ),
    ],
)
def test_public_names(package, names):
    """Check the names exported by the public packages."""
    assert set(package.__all__) == names
    assert all(hasattr(package, name) for name in names)
