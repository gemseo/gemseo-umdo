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
"""A function approximating the Hessian matrix by finite differences."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.problem.database import Database
from gemseo.util.derivative.approximator.forward_differences import ForwardDifferences

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.util.typing import NumberArray
    from gemseo.util.typing import RealArray


class HessianFunction(ArrayFunction):
    """A function approximating the Hessian matrix by finite differences.

    Take an original function and approximate its Hessian with finite differences
    applied to its analytical or approximated Jacobian.
    """

    __jac: Callable[[NumberArray], NumberArray]
    """The function computing the Jacobian."""

    def __init__(self, func: ArrayFunction) -> None:
        """
        Args:
            func: The original function.
        """  # noqa: D205 D212 D415
        self.__jac = (
            func.jac if func.has_jac else ForwardDifferences(func.func).f_gradient
        )
        grad_tag = Database.grad_tag
        super().__init__(
            ForwardDifferences(self._compute_jac).f_gradient,
            f"{grad_tag}{grad_tag}{func.name}",
        )

    def _compute_jac(self, input_data: RealArray) -> RealArray:
        """Compute the Jacobian matrix.

        Args:
            input_data: The input data.

        Returns:
            The Jacobian matrix.
        """
        return self.__jac(input_data).T
