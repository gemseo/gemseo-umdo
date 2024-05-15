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
"""Functions to estimate statistics from an
[UMDOFormulation][gemseo_umdo.formulations.formulation.UMDOFormulation].

The base function is a
[StatisticFunction][gemseo_umdo.formulations._functions.statistic_function.StatisticFunction]
and derives from an
[MDOFunction][gemseo.core.mdofunctions.mdo_function.MDOFunction].
Most of the other _functions derive from
[StatisticFunction][gemseo_umdo.formulations._functions.statistic_function.StatisticFunction]
and are associated with an
[UMDOFormulation][gemseo_umdo.formulations.formulation.UMDOFormulation],
e.g.
[Sampling][gemseo_umdo.formulations.sampling.Sampling].
and
[TaylorPolynomial][gemseo_umdo.formulations.taylor_polynomial.TaylorPolynomial].
The other modules are helpers.
"""  # noqa: D205

from __future__ import annotations
