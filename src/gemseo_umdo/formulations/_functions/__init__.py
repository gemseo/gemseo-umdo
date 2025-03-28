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
"""Functions to estimate statistics from a [BaseUMDOFormulation][gemseo_umdo.formulatio
ns.base_umdo_formulation.BaseUMDOFormulation].

The base function is a
[BaseStatisticFunction][gemseo_umdo.formulations._functions.base_statistic_function.BaseStatisticFunction]
and derives from an
[MDOFunction][gemseo.core.mdo_functions.mdo_function.MDOFunction].
Most of the other _functions derive from
[BaseStatisticFunction][gemseo_umdo.formulations._functions.base_statistic_function.BaseStatisticFunction]
and are associated with an
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation],
e.g.
[Sampling][gemseo_umdo.formulations.sampling.Sampling].
and
[TaylorPolynomial][gemseo_umdo.formulations.taylor_polynomial.TaylorPolynomial].
The other modules are helpers.
"""  # noqa: D205

from __future__ import annotations
