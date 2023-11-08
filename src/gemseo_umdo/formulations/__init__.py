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

An [MDOFormulation][gemseo.core.formulation.MDOFormulation]
defines an [OptimizationProblem][gemseo.algos.opt_problem.OptimizationProblem]
from one or several [MDODiscipline][gemseo.core.discipline.MDODiscipline]s,
a [DesignSpace][gemseo.algos.design_space.DesignSpace],
an objective and constraints.
The objective can be either minimized (default) or maximized.

In the context of deterministic MDO,
the [OptimizationProblem][gemseo.algos.opt_problem.OptimizationProblem]
is handled by a driver
(see [DriverLibrary][gemseo.algos.driver_library.DriverLibrary]),
typically an optimizer
(see [OptimizationLibrary][gemseo.algos.opt.optimization_library.OptimizationLibrary]),
or a design of experiments
(DOE, see [DOELibrary][gemseo.algos.doe.doe_library.DOELibrary]).

In the frame of robust MDO,
the [UMDOFormulation][gemseo_umdo.formulations.formulation.UMDOFormulation]
uses a [MDOFormulation][gemseo.core.formulation.MDOFormulation]
with a [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
defining the uncertain variables
and executes the corresponding
[OptimizationProblem][gemseo.algos.opt_problem.OptimizationProblem]
with a particular DOE.
Then,
it post-processed the associated [Database][gemseo.algos.database.Database]
to estimate the statistics applied to the objective and constraints.

The most common [UMDOFormulation][gemseo_umdo.formulations.formulation.UMDOFormulation]
is [Sampling][gemseo_umdo.formulations.sampling.Sampling],
consisting in estimating the statistics with (quasi) Monte Carlo techniques.
"""

from __future__ import annotations
