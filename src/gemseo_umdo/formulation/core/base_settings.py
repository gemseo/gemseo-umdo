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
"""The base class for the settings of U-MDO formulations."""

from __future__ import annotations

from typing import Any

from gemseo.formulation.core.base_settings import BaseFormulationSettings


class BaseUMDOFormulationSettings(BaseFormulationSettings):
    """The base class for the settings of U-MDO formulations."""

    def __setattr__(self, name: str, value: Any) -> None:
        # An assignment changes the settings in place
        # before the model validators check them
        # and pydantic does not undo it when one of them raises an error,
        # so their state is saved to restore it in that case.
        state = (self.__dict__.copy(), self.__pydantic_fields_set__.copy())
        try:
            super().__setattr__(name, value)
        except Exception:
            object.__setattr__(self, "__dict__", state[0])
            object.__setattr__(self, "__pydantic_fields_set__", state[1])
            raise
