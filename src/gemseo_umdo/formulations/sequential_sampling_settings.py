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

from gemseo.algos.doe.base_n_samples_based_doe_settings import (
    BaseNSamplesBasedDOESettings,  # noqa: TC002
)
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.utils.seeder import SEED
from pydantic import Field
from pydantic import PositiveInt

from gemseo_umdo.formulations.sampling_settings import Sampling_Settings


class SequentialSampling_Settings(Sampling_Settings):  # noqa: N801
    """The settings for the sequential sampling-based U-MDO formulation."""

    _TARGET_CLASS_NAME = "SequentialSampling"

    doe_algo_settings: BaseNSamplesBasedDOESettings = Field(
        default=OT_OPT_LHS_Settings(n_samples=10, seed=SEED),
        description="The DOE settings.",
    )

    initial_n_samples: PositiveInt = Field(
        default=2, description="""The initial sampling size."""
    )

    n_samples_increment: PositiveInt = Field(
        default=1, description="""The increment of the sampling size."""
    )
