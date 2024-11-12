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
"""A discipline adding a random variable to a variable."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import eye
from numpy import ones

from gemseo_umdo.disciplines.base_noiser import BaseNoiser

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


class AdditiveNoiser(BaseNoiser):
    """A discipline adding a random variable to a variable."""

    SHORT_NAME: ClassVar[str] = "+"

    def _run(self, input_data: StrKeyMapping) -> None:
        self.io.update_output_data({
            self._noised_variable_name: (
                self.io.data[self._variable_name]
                + self.io.data[self._uncertain_variable_name]
            )
        })

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        variable_size = self.io.data[self._variable_name].size
        uncertain_variable_size = self.io.data[self._uncertain_variable_name].size
        if uncertain_variable_size == 1:
            u_jacobian = ones((variable_size, 1))
        else:
            u_jacobian = eye(variable_size)
        self.jac = {
            self._noised_variable_name: {
                self._variable_name: eye(variable_size),
                self._uncertain_variable_name: u_jacobian,
            }
        }
