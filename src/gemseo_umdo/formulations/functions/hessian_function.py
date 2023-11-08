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

from gemseo.algos.database import Database
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.derivatives.finite_differences import FirstOrderFD

if TYPE_CHECKING:
    from numpy import ndarray


class HessianFunction(MDOFunction):
    """A function approximating the Hessian matrix by finite differences.

    Take an original function and approximate its Hessian with finite differences
    applied to its analytical or approximated Jacobian.
    """

    __jac: MDOFunction
    """The function computing the Jacobian."""

    def __init__(self, func: MDOFunction) -> None:
        """
        Args:
            func: The original function.
        """  # noqa: D205 D212 D415
        self.__jac = func.jac if func.has_jac else FirstOrderFD(func.func).f_gradient
        grad_tag = Database.GRAD_TAG
        super().__init__(
            FirstOrderFD(self._compute_jac).f_gradient,
            f"{grad_tag}{grad_tag}{func.name}",
        )

    def _compute_jac(self, input_data: ndarray) -> ndarray:
        """Compute the Jacobian matrix.

        Args:
            input_data: The input data.

        Returns:
            The Jacobian matrix.
        """
        return self.__jac(input_data).T
