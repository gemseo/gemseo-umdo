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
"""Settings for the PCE-based U-MDO formulation."""

from __future__ import annotations

from gemseo.mlearning.regression.algos.pce_settings import PCERegressor_Settings
from pydantic import Field
from pydantic import PositiveFloat

from gemseo_umdo.formulations.surrogate_settings import Surrogate_Settings


class PCE_Settings(Surrogate_Settings):  # noqa: N801
    """The settings for the PCE-based U-MDO formulation."""

    _TARGET_CLASS_NAME = "PCE"

    regressor_settings: PCERegressor_Settings = Field(
        default=PCERegressor_Settings(),
        description="The PCE settings.",
    )

    approximate_statistics_jacobians: bool = Field(
        default=False,
        description=(
            "Whether to approximate "
            "the Jacobian of the mean, standard deviation and variance of the PCE "
            "using the technique proposed in Section II.C.3 "
            "of a paper by Mura _et al._[@Mura2020]."
        ),
    )

    differentiation_step: PositiveFloat = Field(
        default=1e-6,
        description=(
            "The differentiation step for the technique "
            "proposed in Section II.C.3 of a paper by Mura _et al._[@Mura2020]."
        ),
    )
