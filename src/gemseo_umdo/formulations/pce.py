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
r"""FCE-based U-MDO formulation.

[PCE][gemseo_umdo.formulations.pce.PCE] is a
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
estimating the statistics of $f(x,U)$ at a point $x$
from the coefficients of a functional chaos expansion (FCE) model

$$\hat{f}_x(U)=\alpha_0 + \sum_{1\geq i\leq P}\alpha_i\Phi_i(U)$$

built over the uncertain space.

E.g.

$$\mathbb{E}[f(x,U)] \approx \alpha_0$$

and

$$\mathbb{V}[f(x,U)] \approx \sum_{1\geq i\leq P}\alpha_i^2.$$

The best-known FCE is the polynomial chaos expansion (PCE),
hence the name of the U-MDO formulation.
"""

from __future__ import annotations

from typing import ClassVar

from gemseo_umdo.formulations._functions.statistic_function_for_pce import (
    StatisticFunctionForPCE,
)
from gemseo_umdo.formulations._statistics.pce.factory import PCEEstimatorFactory
from gemseo_umdo.formulations.pce_settings import PCE_Settings
from gemseo_umdo.formulations.surrogate import Surrogate


class PCE(Surrogate):
    """FCE-based U-MDO formulation.

    !!! note "DOE algorithms"
        This formulation uses a DOE algorithm;
        read the
        [GEMSEO documentation](https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html).
        for more information about the available DOE algorithm names and options.
    """

    Settings: ClassVar[type[PCE_Settings]] = PCE_Settings

    _STATISTIC_FACTORY: ClassVar[PCEEstimatorFactory] = PCEEstimatorFactory()

    _STATISTIC_FUNCTION_CLASS: ClassVar[type[StatisticFunctionForPCE] | None] = (
        StatisticFunctionForPCE
    )
