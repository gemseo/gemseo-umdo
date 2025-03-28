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
"""Settings for the surrogate-based U-MDO formulation."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003
from collections.abc import Sequence  # noqa: TC003

from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,  # noqa: TC002
)
from gemseo.mlearning.regression.algos.rbf_settings import RBFRegressor_Settings
from gemseo.utils.seeder import SEED
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt

from gemseo_umdo.formulations.base_sampling_settings import BaseSamplingSettings


class Surrogate_Settings(BaseSamplingSettings):  # noqa: N801
    """The settings for the surrogate-based U-MDO formulation."""

    _TARGET_CLASS_NAME = "Surrogate"

    regressor_settings: BaseRegressorSettings = Field(
        default=RBFRegressor_Settings(), description="The regressor settings."
    )

    regressor_n_samples: PositiveInt = Field(
        default=10000,
        description="""The number of Monte Carlo samples
to estimate the statistics from the regressor.""",
    )

    regressor_sampling_seed: NonNegativeInt = Field(
        default=SEED, description="The seed of the Monte Carlo sampler."
    )

    quality_name: str = Field(
        default="R2Measure",
        description="The name of the measure to assess the quality of the regressor.",
    )

    quality_threshold: float | Mapping[str, float | Sequence[float]] = Field(
        default=0.9,
        description="The learning quality threshold below which a warning is logged.",
    )

    quality_cv_compute: bool = Field(
        default=True,
        description="Whether to estimate the quality by cross-validation (CV).",
    )

    quality_cv_n_folds: PositiveInt = Field(
        default=5, description="The number of folds in the case of the CV technique."
    )

    quality_cv_randomize: bool = Field(
        default=True,
        description="""Whether to shuffle the samples
before dividing them in folds in the case of the CV technique.""",
    )

    quality_cv_seed: NonNegativeInt | None = Field(
        default=None,
        description="""The seed of the pseudo-random number generator.

If ``None``,
an unpredictable generator is used.""",
    )

    quality_cv_threshold: float | Mapping[str, float | Sequence[float]] = Field(
        default=0.8,
        description="The CV quality threshold below which a warning is logged.",
    )
