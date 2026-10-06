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

    from gemseo.util.typing import StrKeyMapping


class AdditiveNoiser(BaseNoiser):
    """A discipline adding a random variable to a variable."""

    SHORT_NAME: ClassVar[str] = "+"

    def _run(self, input_data: StrKeyMapping) -> None:
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data, get(name) or get_merged_data()  # noqa: E501
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
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data, get(name) or get_merged_data()  # noqa: E501
        variable_size = self.io.data[self._variable_name].size
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data, get(name) or get_merged_data()  # noqa: E501
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
