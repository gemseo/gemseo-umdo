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

A [BaseMDOFormulation][gemseo.formulations.base_mdo_formulation.BaseMDOFormulation]
defines an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
from one or several [Disciplines][gemseo.core.discipline.discipline.Discipline],
a [DesignSpace][gemseo.algos.design_space.DesignSpace],
an objective and constraints.
The objective can be either minimized (default) or maximized.

In the context of deterministic MDO,
the [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
is handled by a driver
(see [DriverLibrary][gemseo.algos.base_driver_library.BaseDriverLibrary]),
typically an optimizer
(see
[OptimizationLibrary][gemseo.algos.opt.base_optimization_library.BaseOptimizationLibrary]),
or a design of experiments
(DOE, see [DOELibrary][gemseo.algos.doe.base_doe_library.BaseDOELibrary]).

In the frame of U-MDO,
the
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
uses a [BaseMDOFormulation][gemseo.formulations.base_mdo_formulation.BaseMDOFormulation]
with a [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
defining the uncertain variables
and executes the corresponding
[OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
with a particular DOE.
Then,
it post-processed the associated [Database][gemseo.algos.database.Database]
to estimate the statistics applied to the objective and constraints.

The most common
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
is [Sampling][gemseo_umdo.formulations.sampling.Sampling],
consisting in estimating the statistics with (quasi) Monte Carlo techniques.
"""

from __future__ import annotations

from gemseo.algos.doe.factory import DOELibraryFactory
from strenum import StrEnum

DOE_ALGO_NAMES = StrEnum("DOE_ALGO_NAMES", DOELibraryFactory().algorithms)
"""The names of the available DOE algorithms."""
