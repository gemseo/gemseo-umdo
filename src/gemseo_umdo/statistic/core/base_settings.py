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
"""The base settings of the statistics of the U-MDO formulations.

A statistic is defined by a settings model `X_Settings`
deriving from
[BaseStatisticSettings][gemseo_umdo.statistic.core.base_settings.BaseStatisticSettings]
and by one estimator class `X` per estimation technique
supporting this statistic,
found by the factory of statistic estimators of this technique.
"""

from __future__ import annotations

from typing import Any
from typing import ClassVar
from typing import Final
from typing import Self
from weakref import WeakValueDictionary

from gemseo.uncertainty.statistic.core.base import BaseStatistics
from gemseo.util.file_path_manager import FilePathManager
from gemseo.util.pydantic import BaseSettings

_target_class_name_attribute: Final[str] = "_BaseSettings__target_class_name"
"""The class attribute where GEMSEO stores the target class name of settings."""

_name_to_class: Final[WeakValueDictionary[str, type[BaseStatisticSettings]]] = (
    WeakValueDictionary()
)
"""The settings classes of the statistics bound to their names.

A class is registered at its definition
and a later definition of a class with the same name replaces it,
e.g. when a notebook cell defining a class is run again.
The references are weak so that this registry does not keep a class alive.
"""


class BaseStatisticSettings(BaseSettings):
    """The base settings of a statistic.

    The name of a subclass must end with `_Settings`,
    e.g. `Margin_Settings`,
    the prefix being the name of the classes of the estimators of this statistic,
    e.g. `Margin`.
    """

    _name_field_names: ClassVar[tuple[str, ...]] = ()
    """The names of the fields always used in the name of the statistic.

    The other fields are used only when they are set explicitly.
    """

    def make_conservative(self, from_below: bool) -> Self:
        """Return the settings making the statistic conservative.

        By default, the settings are returned unchanged.

        Args:
            from_below: Whether the statistic must be conservative from below,
                e.g. for an objective to maximize or a positive inequality
                constraint.
                Otherwise,
                it must be conservative from above,
                e.g. for an objective to minimize or a negative inequality
                constraint.

        Returns:
            The settings making the statistic conservative.
        """
        return self

    def get_name_parameters(self) -> dict[str, Any]:
        """Return the parameters of the statistic to be used in its name.

        Returns:
            The parameters of the statistic to be used in its name.
        """
        return {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name in self._name_field_names or name in self.model_fields_set
        }

    def compute_expression(self, output_name: str, show_name: bool = False) -> str:
        """Return the string representation of the statistic of an output.

        Args:
            output_name: The name of the output.
            show_name: Whether to show the name of the statistic.

        Returns:
            The string representation of the statistic of the output.
        """
        return BaseStatistics.compute_expression(
            output_name,
            FilePathManager.to_snake_case(self.target_class_name),
            show_name=show_name,
            **self.get_name_parameters(),
        )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        # GEMSEO sets the target class name of a class
        # when the name of this class ends with the suffix of the settings,
        # e.g. Margin for Margin_Settings;
        # the other classes, e.g. intermediate base classes, are not statistics.
        name = vars(cls).get(_target_class_name_attribute)
        if name:
            _name_to_class[name] = cls

    @classmethod
    def from_name(cls, name: str) -> BaseStatisticSettings:
        """Return the default settings of a statistic from its name.

        When several settings classes are defined for the same statistic,
        the most recently defined one is used.

        Args:
            name: The name of the statistic, e.g. `"Margin"`.

        Returns:
            The default settings of the statistic.

        Raises:
            ValueError: When no settings class is defined for this statistic.
        """
        settings_class = _name_to_class.get(name)
        if settings_class is None:
            msg = (
                f"The statistic '{name}' has no settings class; "
                f"available statistics: {', '.join(sorted(_name_to_class))}."
            )
            raise ValueError(msg)

        return settings_class()
