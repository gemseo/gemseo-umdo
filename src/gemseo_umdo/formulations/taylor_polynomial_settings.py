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
"""Settings for the U-MDO formulation based on Taylor polynomials."""

from __future__ import annotations

from gemseo.algos.optimization_problem import OptimizationProblem
from pydantic import Field

from gemseo_umdo.formulations.base_umdo_formulation_settings import (
    BaseUMDOFormulationSettings,
)


class TaylorPolynomial_Settings(BaseUMDOFormulationSettings):  # noqa: N801
    """The settings for the U-MDO formulation based on Taylor polynomials."""

    _TARGET_CLASS_NAME = "TaylorPolynomial"

    differentiation_method: OptimizationProblem.DifferentiationMethod = Field(
        default=OptimizationProblem.DifferentiationMethod.USER_GRAD,
        description="The type of method to compute the gradients.",
    )

    second_order: bool = Field(
        default=False,
        description="Whether to use second-order Taylor polynomials "
        "instead of first-order Taylor polynomials.",
    )
