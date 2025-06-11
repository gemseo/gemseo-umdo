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
"""Settings for the control variate-based U-MDO formulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos.doe.base_doe_settings import BaseDOESettings  # noqa: TC002
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,  # noqa: TC002
)
from gemseo.utils.seeder import SEED
from pydantic import Field
from pydantic import model_validator

from gemseo_umdo.formulations.base_surrogate_settings import (
    BaseSurrogateWithMCStatistics_Settings,
)
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings

if TYPE_CHECKING:
    from typing_extensions import Self


class ControlVariate_Settings(  # noqa: N801
    Sampling_Settings, BaseSurrogateWithMCStatistics_Settings
):
    """The settings for the control variate-based U-MDO formulation.

    The DOE settings for creating the training dataset are defined
    using ``regressor_doe_algo_settings``.
    The statistics of the surrogate are estimated by Monte Carlo sampling
    using ``regressor_n_samples`` and ``regressor_sampling_seed``.
    """

    _TARGET_CLASS_NAME = "ControlVariate"

    regressor_doe_algo_settings: BaseDOESettings = Field(
        default=OT_OPT_LHS_Settings(n_samples=10, seed=SEED + 1),
        description=(
            """The DOE settings for creating the training dataset for the regressor.

This field is ignored when ``regressor_settings`` is ``None``.
"""
        ),
    )

    regressor_settings: BaseRegressorSettings | None = Field(
        default=None,
        description="""The settings of the regressor used by the control variates.

If ``None``, the control variates use first-order Taylor polynomials.""",
    )

    @model_validator(mode="after")
    def __validate_seeds(self) -> Self:
        """Validate the seeds."""
        if (
            "seed" in self.doe_algo_settings.model_fields
            and "seed" in self.regressor_doe_algo_settings.model_fields
            and self.doe_algo_settings._TARGET_CLASS_NAME
            == self.regressor_doe_algo_settings._TARGET_CLASS_NAME
            and self.doe_algo_settings.seed == self.regressor_doe_algo_settings.seed
        ):
            msg = (
                "The seed for sampling and the seed for creating the training dataset "
                f"must be different; both are {self.doe_algo_settings.seed}."
            )
            raise ValueError(msg)
        return self
