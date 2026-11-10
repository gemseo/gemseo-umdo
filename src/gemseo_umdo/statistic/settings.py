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
"""The settings of the statistics of the U-MDO formulations.

A statistic is defined by a settings model `X_Settings`
deriving from
[BaseStatisticSettings][gemseo_umdo.statistic.settings.BaseStatisticSettings]
and by one estimator class `X` per estimation technique
supporting this statistic,
found by the factory of statistic estimators of this technique.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from typing import ClassVar
from typing import Self

from gemseo.uncertainty.statistic.core.base import BaseStatistics
from gemseo.util.file_path_manager import FilePathManager
from gemseo.util.pydantic import BaseSettings
from numpy import array
from numpy import quantile
from pydantic import Field
from pydantic import field_validator


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
                    name = subclass.model_construct().target_class_name
                    if name:
                        name_to_class.setdefault(name, subclass)
                    subclasses.append(subclass)

            classes = subclasses

        return name_to_class


class Mean_Settings(BaseStatisticSettings):  # noqa: N801
    """The settings of the mean."""


class Variance_Settings(BaseStatisticSettings):  # noqa: N801
    """The settings of the variance."""


class StandardDeviation_Settings(BaseStatisticSettings):  # noqa: N801
    """The settings of the standard deviation."""


class Margin_Settings(BaseStatisticSettings):  # noqa: N801
    r"""The settings of the margin $\mathbb{E}[\varphi]+\kappa\mathbb{S}[\varphi]$."""

    _name_field_names: ClassVar[tuple[str, ...]] = ("factor",)

    factor: float = Field(
        2.0,
        description=r"""The factor $\kappa$ related to the standard deviation.

For an objective or an inequality constraint,
its sign is changed if necessary to make the margin conservative.""",
    )

    @field_validator("factor")
    @classmethod
    def _check_factor(cls, factor: float) -> float:
        """Check that the factor is non-zero.

        Args:
            factor: The factor.

        Returns:
            The factor.

        Raises:
            ValueError: When the factor is zero.
        """
        if factor == 0:
            msg = (
                "The factor of a margin must be non-zero; "
                "a margin with a zero factor is the mean. "
                "Use the statistic 'Mean' instead."
            )
            raise ValueError(msg)

        return factor

    def make_conservative(self, from_below: bool) -> Self:  # noqa: D102
        factor = abs(self.factor)
        if from_below:
            factor *= -1

        return self.model_copy(update={"factor": factor})


class Probability_Settings(BaseStatisticSettings):  # noqa: N801
    """The settings of the probability that the output exceeds a threshold."""

    threshold: float = Field(
        0.0, description="The threshold against which the probability is estimated."
    )

    greater: bool = Field(
        True,
        description="""Whether to compute the probability of exceeding the threshold.

Otherwise, compute the probability of being below the threshold.""",
    )


class Quantile_Settings(BaseStatisticSettings):  # noqa: N801
    r"""The settings of the quantile."""

    _name_field_names: ClassVar[tuple[str, ...]] = ("level",)

    level: float = Field(
        0.95,
        gt=0,
        lt=1,
        description=r"""The level $\alpha$ of the quantile, in $]0,1[$.

For an objective or an inequality constraint,
it is replaced by $1-\alpha$ if necessary to make the quantile conservative.""",
    )

    method: str = Field(
        "linear",
        description="""The name of the method of
[numpy.quantile][numpy.quantile] to estimate the quantile.""",
    )

    @field_validator("method")
    @classmethod
    def _check_method(cls, method: str) -> str:
        """Check that the method is a method of `numpy.quantile`.

        Args:
            method: The name of the method.

        Returns:
            The name of the method.

        Raises:
            ValueError: When the method is not a method of `numpy.quantile`.
        """
        # The method is checked with a cheap call to numpy.quantile
        # so that a wrong method fails before any sampling,
        # as NumPy does not expose its list of methods.
        try:
            quantile(array([0.0, 1.0]), 0.5, method=method)
        except ValueError as error:
            msg = (
                "The method of the quantile must be a method of numpy.quantile; "
                f"got '{method}'."
            )
            raise ValueError(msg) from error

        return method

    def make_conservative(self, from_below: bool) -> Self:  # noqa: D102
        # The conversion to float handles NumPy scalars,
        # whose repr is not a valid decimal string, e.g. np.float64(0.9).
        level = float(self.level)
        # The decimal arithmetic avoids names like Q[f; 0.09999999999999998]
        # for the complementary level of 0.9.
        complementary_level = float(1 - Decimal(repr(level)))
        level = (
            min(level, complementary_level)
            if from_below
            else max(level, complementary_level)
        )
        return self.model_copy(update={"level": level})
