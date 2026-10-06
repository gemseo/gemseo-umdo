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

    from gemseo.util.typing import StrKeyMapping


class MultiplicativeNoiser(BaseNoiser):
    """A discipline multiplying a variable by a random variable plus one."""

    SHORT_NAME: ClassVar[str] = "*"

    def _run(self, input_data: StrKeyMapping) -> None:
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data, get(name) or get_merged_data()  # noqa: E501
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
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data, get(name) or get_merged_data()  # noqa: E501
        uncertain_variable_value = self.io.data[self._uncertain_variable_name]
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data, get(name) or get_merged_data()  # noqa: E501
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
