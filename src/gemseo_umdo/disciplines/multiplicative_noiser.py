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
"""A discipline multiplying a variable by a random variable plus one."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import diag
from numpy import eye
from numpy import newaxis

from gemseo_umdo.disciplines.base_noiser import BaseNoiser

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


class MultiplicativeNoiser(BaseNoiser):
    """A discipline multiplying a variable by a random variable plus one."""

    SHORT_NAME: ClassVar[str] = "*"

    def _run(self, input_data: StrKeyMapping) -> None:
        self.io.update_output_data({
            self._noised_variable_name: (
                self.io.data[self._variable_name]
                * (1 + self.io.data[self._uncertain_variable_name])
            )
        })

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        uncertain_variable_value = self.io.data[self._uncertain_variable_name]
        variable_value = self.io.data[self._variable_name]
        if uncertain_variable_value.size == 1:
            x_jacobian = eye(variable_value.size) * (1 + uncertain_variable_value)
            u_jacobian = variable_value[:, newaxis]
        else:
            x_jacobian = diag(1 + uncertain_variable_value)
            u_jacobian = diag(variable_value)

        self.jac = {
            self._noised_variable_name: {
                self._variable_name: x_jacobian,
                self._uncertain_variable_name: u_jacobian,
            }
        }
