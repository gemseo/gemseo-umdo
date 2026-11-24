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
"""Settings for the sequential sampling-based U-MDO formulation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from gemseo.doe import OT_OPT_LHS_Settings
from gemseo.doe.core.base_n_samples_based_doe_settings import (
    BaseNSamplesBasedDOESettings,
)
from gemseo.util.seeder import seed
from pydantic import Field
from pydantic import PositiveInt
from pydantic import model_validator

from gemseo_umdo.formulation.sampling_settings import Sampling_Settings

if TYPE_CHECKING:
    from typing import Any
    from typing import Self

    from pydantic import ModelWrapValidatorHandler


class SequentialSampling_Settings(Sampling_Settings):  # noqa: N801
    """The settings for the sequential sampling-based U-MDO formulation."""

    doe_algo_settings: BaseNSamplesBasedDOESettings = Field(
        default=OT_OPT_LHS_Settings(n_samples=10, seed=seed),
        description="The DOE settings.",
    )

    initial_n_samples: PositiveInt = Field(
        default=2,
        description="""The initial sampling size.

The estimation of a variance, a standard deviation
or a margin
requires at least 2 samples.""",
    )

    n_samples_increment: PositiveInt | Callable[[PositiveInt], PositiveInt] = Field(
        default=1,
        description=(
            """Either the increment of the sampling size
or a function computing this increment from the current sampling size.
In the first case, the increment will be replaced by such a function."""
        ),
    )

    @model_validator(mode="wrap")
    @classmethod
    def __check_initial_n_samples(
        cls, data: Any, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        """Check that the initial sampling size is at most the maximum one.

        Args:
            data: The data to validate,
                or the settings themselves in the case of an assignment.
            handler: The validator of the data.

        Returns:
            The validated settings.

        Raises:
            ValueError: When the initial sampling size is greater than
                the number of samples of the DOE settings.
        """
        # An assignment changes the settings in place before this check,
        # so their state is saved to restore it when the check fails.
        state = (
            (data.__dict__.copy(), data.__pydantic_fields_set__.copy())
            if isinstance(data, cls)
            else None
        )
        settings = handler(data)
        n_samples = settings.doe_algo_settings.n_samples
        if settings.initial_n_samples > n_samples:
            if state is not None:
                object.__setattr__(settings, "__dict__", state[0])
                object.__setattr__(settings, "__pydantic_fields_set__", state[1])

            msg = (
                f"The initial sampling size ({settings.initial_n_samples}) "
                f"must be less than or equal to "
                f"the maximum sampling size ({n_samples})."
            )
            raise ValueError(msg)

        return settings

    @model_validator(mode="after")
    def __make_n_samples_increment_callable(self) -> Self:
        """Make n_samples_increment callable."""
        if isinstance(self.n_samples_increment, int):
            self.n_samples_increment = _DefaultIncrementor(self.n_samples_increment)

        return self


class _DefaultIncrementor:
    """Sampling size incrementor."""

    __n_samples_increment: PositiveInt
    """The increment of the sampling size."""

    def __init__(self, n_samples_increment: PositiveInt) -> None:
        """
        Args:
            n_samples_increment: The increment of the sampling size.
        """  # noqa: D205 D212 D415
        self.__n_samples_increment = n_samples_increment

    def __call__(self, n_samples: PositiveInt) -> PositiveInt:
        """Compute the increment of the sampling size.

        Args:
            n_samples: The current number of samples.

        Returns:
            The increment of the sampling size.
        """
        return self.__n_samples_increment
