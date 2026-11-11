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

from gemseo.uncertainty.statistic.core.base import BaseStatistics
from gemseo.util.file_path_manager import FilePathManager
from gemseo.util.pydantic import BaseSettings

_settings_suffix: Final[str] = "_Settings"
"""The suffix of the name of a settings class of a statistic."""


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
        A derived class changing the settings must return a deep copy of them.

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
    def from_name(cls, name: str) -> BaseStatisticSettings:
        """Return the default settings of a statistic from its name.

        Args:
            name: The name of the statistic, e.g. `"Margin"`.

        Returns:
            The default settings of the statistic.

        Raises:
            ValueError: When no settings class is defined for this statistic.
        """
        name_to_class = cls.__get_name_to_class()
        settings_class = name_to_class.get(name)
        if settings_class is None:
            msg = (
                f"The statistic '{name}' has no settings class; "
                f"available statistics: {', '.join(sorted(name_to_class))}."
            )
            raise ValueError(msg)

        return settings_class()

    @classmethod
    def __get_name_to_class(cls) -> dict[str, type[BaseStatisticSettings]]:
        """Return the settings classes of the statistics bound to their names.

        The name of a statistic is read from the name of its settings class,
        without instantiating this class.
        A class closer to `BaseStatisticSettings` takes precedence.

        Returns:
            The settings classes of the statistics bound to their names.
        """
        name_to_class = {}
        classes = [BaseStatisticSettings]
        while classes:
            subclasses = []
            for class_ in classes:
                for subclass in class_.__subclasses__():
                    class_name = subclass.__name__
                    if class_name.endswith(_settings_suffix):
                        name = class_name.removesuffix(_settings_suffix)
                        name_to_class.setdefault(name, subclass)
                    subclasses.append(subclass)

            classes = subclasses

        return name_to_class
