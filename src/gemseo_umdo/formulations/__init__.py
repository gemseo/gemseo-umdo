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

A :class:`~gemseo.core.formulation.MDOFormulation`
defines an :class:`~gemseo.algos.opt_problem.OptimizationProblem`
from one or several :class:`~gemseo.core.discipline.MDODiscipline`s,
a :class:`~gemseo.algos.design_space.DesignSpace`,
an objective and constraints.
The objective can be either minimized (default) or maximized.

In the context of deterministic MDO,
the :class:`~gemseo.algos.opt_problem.OptimizationProblem`
is handled by a driver (see :class:`~gemseo.algos.driver_lib.DriverLib`),
typically an optimizer (see :class:`~gemseo.algos.opt.opt_lib.OptimizationLibrary`),
or a design of experiments (DOE, see :class:`~gemseo.algos.doe.doe_lib.DOELibrary`).

In the frame of robust MDO,
the :class:`~gemseo_umdo.formulations.formulation.UMDOFormulation`
uses a :class:`~gemseo.core.formulation.MDOFormulation`
with a :class:`~gemseo.algos.parameter_space.ParameterSpace`
defining the uncertain variables
and executes the corresponding :class:`~gemseo.algos.opt_problem.OptimizationProblem`
with a particular DOE.
Then,
it post-processed the associated :class:`~gemseo.algos.database.Database`
to estimate the statistics applied to the objective and constraints.

The most common :class:`~gemseo_umdo.formulations.formulation.UMDOFormulation`
is :class:`~gemseo_umdo.formulations.sampling.Sampling`,
consisting in estimating the statistics with (quasi) Monte Carlo techniques.
"""
from __future__ import annotations
