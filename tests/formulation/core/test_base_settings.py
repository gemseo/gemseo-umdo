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
from __future__ import annotations

import pytest
from gemseo.doe import CustomDOE_Settings
from gemseo.doe import OT_OPT_LHS_Settings
from gemseo.util.seeder import seed
from numpy import array
from pydantic import ValidationError

from gemseo_umdo.formulation.control_variate_settings import ControlVariate_Settings
from gemseo_umdo.formulation.sampling_settings import Sampling_Settings
from gemseo_umdo.formulation.sequential_sampling_settings import (
    SequentialSampling_Settings,
)


@pytest.mark.parametrize(
    ("create_settings", "name", "value"),
    [
        (
            lambda: Sampling_Settings(
                doe_algo_settings=CustomDOE_Settings(samples=array([[0.0]]))
            ),
            "n_samples",
            3,
        ),
        (
            ControlVariate_Settings,
            "regressor_doe_algo_settings",
            OT_OPT_LHS_Settings(n_samples=10, seed=seed),
        ),
        (
            lambda: SequentialSampling_Settings(initial_n_samples=5, n_samples=10),
            "n_samples",
            3,
        ),
        (
            lambda: SequentialSampling_Settings(initial_n_samples=5, n_samples=10),
            "initial_n_samples",
            11,
        ),
    ],
    ids=[
        "sampling_n_samples",
        "control_variate_seeds",
        "sequential_sampling_n_samples",
        "sequential_sampling_initial_n_samples",
    ],
)
def test_rejected_assignment_leaves_settings_unchanged(create_settings, name, value):
    """Check that an assignment rejected by a model validator changes nothing."""
    settings = create_settings()
    old_value = getattr(settings, name)
    doe_algo_settings = settings.doe_algo_settings
    fields_set = settings.model_fields_set.copy()
    with pytest.raises(ValidationError):
        setattr(settings, name, value)

    assert getattr(settings, name) is old_value
    assert settings.doe_algo_settings is doe_algo_settings
    assert settings.model_fields_set == fields_set
