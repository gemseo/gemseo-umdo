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
"""Formulations for multidisciplinary design problems under uncertainty.

A [BaseMDOFormulation][gemseo.formulation.core.base_mdo.BaseMDOFormulation]
defines an [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem]
from one or several [Disciplines][gemseo.core.discipline.discipline.Discipline],
a [DesignSpace][gemseo.space.design.DesignSpace],
an objective and constraints.
The objective can be either minimized (default) or maximized.

In the context of deterministic MDO,
the [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem]
is handled by a driver
(see [DriverLibrary][gemseo.core.algorithm.base_driver_library.BaseDriverLibrary]),
typically an optimizer
(see
[OptimizationLibrary][gemseo.optimization.core.base_optimization_library.BaseOptimizationLibrary]),
or a design of experiments
(DOE, see [DOELibrary][gemseo.doe.core.base_doe_library.BaseDOELibrary]).

In the frame of U-MDO,
the
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
uses a [BaseMDOFormulation][gemseo.formulation.core.base_mdo.BaseMDOFormulation]
with a [RandomSpace][gemseo.space.random.RandomSpace]
defining the uncertain variables
and executes the corresponding
[OptimizationProblem][gemseo.optimization.problem.OptimizationProblem]
with a particular DOE.
Then,
it post-processed the associated [Database][gemseo.core.problem.database.Database]
to estimate the statistics of the outputs defining the objective and constraints.

The most common
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
is [Sampling][gemseo_umdo.formulations.sampling.Sampling],
consisting in estimating the statistics with (quasi) Monte Carlo techniques.
"""

from __future__ import annotations

from enum import StrEnum

from gemseo.doe.factory import doe_library_factory

# Only settings modules are imported,
# so that importing this package does not import the U-MDO formulations.
from gemseo_umdo.formulations.control_variate_settings import ControlVariate_Settings
from gemseo_umdo.formulations.pce_settings import PCE_Settings
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.formulations.sequential_sampling_settings import (
    SequentialSampling_Settings,
)
from gemseo_umdo.formulations.statistic_settings import BaseStatisticSettings
from gemseo_umdo.formulations.statistic_settings import Margin_Settings
from gemseo_umdo.formulations.statistic_settings import Mean_Settings
from gemseo_umdo.formulations.statistic_settings import Probability_Settings
from gemseo_umdo.formulations.statistic_settings import Quantile_Settings
from gemseo_umdo.formulations.statistic_settings import StandardDeviation_Settings
from gemseo_umdo.formulations.statistic_settings import Variance_Settings
from gemseo_umdo.formulations.surrogate_settings import Surrogate_Settings
from gemseo_umdo.formulations.taylor_polynomial_settings import (
    TaylorPolynomial_Settings,
)

DOEAlgorithmName = StrEnum(
    "DOEAlgorithmName", {name: name for name in doe_library_factory.algorithms}
)
"""The names of the available DOE algorithms."""

__all__ = [
    "BaseStatisticSettings",
    "ControlVariate_Settings",
    "DOEAlgorithmName",
    "Margin_Settings",
    "Mean_Settings",
    "PCE_Settings",
    "Probability_Settings",
    "Quantile_Settings",
    "Sampling_Settings",
    "SequentialSampling_Settings",
    "StandardDeviation_Settings",
    "Surrogate_Settings",
    "TaylorPolynomial_Settings",
    "Variance_Settings",
]
