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
from gemseo.util.testing.helper import assert_exception
from numpy import array
from pydantic import ValidationError

from gemseo_umdo.formulation.sampling_settings import Sampling_Settings


@pytest.mark.parametrize(
    ("doe_seed", "n_samples", "expected_seed", "expected_n_samples"),
    [
        (None, None, seed, 3),
        (2, 7, 2, 7),
        (None, 7, seed, 7),
        (2, None, 2, 3),
    ],
    ids=["seed", "n_samples", "seed_and_n_samples", "nothing"],
)
def test_doe_algo_settings(doe_seed, n_samples, expected_seed, expected_n_samples):
    """Check the DOE settings set from the settings without changing the user's."""
    doe_algo_settings = OT_OPT_LHS_Settings(n_samples=3, seed=doe_seed)
    settings = Sampling_Settings(
        doe_algo_settings=doe_algo_settings, n_samples=n_samples
    )
    assert settings.doe_algo_settings.seed == expected_seed
    assert settings.doe_algo_settings.n_samples == expected_n_samples
    assert doe_algo_settings.seed == doe_seed
    assert doe_algo_settings.n_samples == 3


def test_n_samples_without_doe_n_samples(snapshot):
    """Check the error when the DOE settings have no number of samples."""
    with assert_exception(ValidationError, snapshot):
        Sampling_Settings(
            doe_algo_settings=CustomDOE_Settings(samples=array([[0.0]])), n_samples=3
        )
