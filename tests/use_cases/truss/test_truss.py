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
from gemseo.utils.testing.helpers import image_comparison
from numpy import array
from numpy import float64
from numpy import ndarray
from numpy.testing import assert_allclose

from gemseo_umdo.use_cases.truss.bar import Bar
from gemseo_umdo.use_cases.truss.bars import Bars
from gemseo_umdo.use_cases.truss.forces import Forces
from gemseo_umdo.use_cases.truss.model import TrussModel


@pytest.fixture(scope="module")
def truss_model() -> TrussModel:
    """A truss model."""
    return TrussModel()


def test_compute_default(truss_model):
    """Check TrussModel.compute with default values."""
    v1, displacements = truss_model.compute()
    assert isinstance(v1, float64)
    assert isinstance(displacements, ndarray)
    assert_allclose(v1, -0.077836, atol=1e-6)
    assert_allclose(
        displacements,
        array([
            [0.0, 0.0],
            [0.00142857, -0.04006768],
            [0.0047619, -0.06791782],
            [0.00904762, -0.07783612],
            [0.01333333, -0.06791782],
            [0.01666667, -0.04006768],
            [0.01809524, 0.0],
            [0.01738095, -0.02142156],
            [0.015, -0.05633285],
            [0.01119048, -0.07569326],
            [0.00690476, -0.07569326],
            [0.00309524, -0.05633285],
            [0.00071429, -0.02142156],
        ]),
        atol=1e-6,
    )


def test_compute_custom(truss_model):
    """Check TrussModel.compute with custom values."""
    bar = Bar(area=1.5e-3)
    bars = Bars(**{f"bar_{i}": bar for i in range(23)})
    forces = Forces(**{f"force_{i}": 5.2e4 for i in range(7, 13)})
    v1, displacements = truss_model.compute(bars=bars, forces=forces)
    assert_allclose(v1, -0.099528, atol=1e-6)
    assert_allclose(
        displacements,
        array([
            [0.0, 0.0],
            [0.00198095, -0.05089137],
            [0.00660317, -0.08670873],
            [0.01254603, -0.09952828],
            [0.01848889, -0.08670873],
            [0.02311111, -0.05089137],
            [0.02509206, 0.0],
            [0.02410159, -0.02690308],
            [0.0208, -0.07157808],
            [0.01551746, -0.09655685],
            [0.0095746, -0.09655685],
            [0.00429206, -0.07157808],
            [0.00099048, -0.02690308],
        ]),
        atol=1e-6,
    )


@image_comparison(["default"])
def test_plot_default(truss_model):
    """Check TrussModel.plot with default values."""
    truss_model.plot(show=False)


@image_comparison(["displacements"])
def test_plot_displacements(truss_model):
    """Check TrussModel.plot with displacements."""
    _, displacements = truss_model.compute()
    truss_model.plot(displacements=displacements, show=False)


@image_comparison(["displacements_scale"])
def test_plot_displacements_scale(truss_model):
    """Check TrussModel.plot with displacements and scaling."""
    _, displacements = truss_model.compute()
    truss_model.plot(displacements=displacements, scale=10, show=False)
