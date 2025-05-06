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
from numpy import array
from numpy.testing import assert_allclose
from openturns import Gumbel
from openturns import LogNormal

from gemseo_umdo.use_cases.truss.uncertain_space import TrussUncertainSpace


def test_helpers():
    """Check protected methods used as helpers."""
    assert TrussUncertainSpace._to_lognormal(1, 2) == pytest.approx((
        -0.8047189562170503,
        1.2686362411795196,
    ))
    assert TrussUncertainSpace._to_gumbel(1, 2) == pytest.approx((
        1.5593936024673523,
        0.09990241871981964,
    ))


@pytest.mark.parametrize("use_different_bars", [False, True])
@pytest.mark.parametrize("factor", [1, 1.5])
def test_default(factor, use_different_bars):
    """Check the default value of TrussUncertainSpace."""
    kwargs = {} if factor == 1 else {"factor": factor}
    if use_different_bars:
        kwargs["use_different_bars"] = use_different_bars
        n1 = 11
        n2 = 12
        e1_names = [f"E1_{i}" for i in range(1, 12)]
        e2_names = [f"E2_{i}" for i in range(1, 13)]
        a1_names = [f"A1_{i}" for i in range(1, 12)]
        a2_names = [f"A2_{i}" for i in range(1, 13)]
    else:
        e1_names = ["E1"]
        e2_names = ["E2"]
        a1_names = ["A1"]
        a2_names = ["A2"]
        n1 = 1
        n2 = 1

    space = TrussUncertainSpace(**kwargs)
    assert [type(m.distribution) for m in space.distribution.marginals] == [
        *[LogNormal] * 2 * (n1 + n2),
        *[Gumbel] * 6,
    ]
    assert space.uncertain_variables == [
        *a1_names,
        *a2_names,
        *e1_names,
        *e2_names,
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
    ]
    assert_allclose(
        space.distribution.mean,
        array([
            *[2e-03] * n1,
            *[1e-03] * n2,
            *[2.1e11] * n1,
            *[2.1e11] * n2,
            *[5e04] * 6,
        ]),
        rtol=1e-6,
    )
    assert_allclose(
        space.distribution.standard_deviation,
        factor
        * array([
            *[2e-04] * n1,
            *[2e-04] * n2,
            *[2.1e10] * n1,
            *[2.1e10] * n2,
            *[7.5e03] * 6,
        ]),
        rtol=1e-6,
    )
