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

from typing import TYPE_CHECKING

import pytest
from gemseo.discipline import AnalyticDiscipline
from gemseo.discipline import DisciplineChain
from gemseo.formulation.mdf import MDF
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from gemseo.uncertainty.distribution import SPNormalDistribution_Settings

from gemseo_umdo.formulations.control_variate_settings import ControlVariate_Settings
from gemseo_umdo.formulations.pce_settings import PCE_Settings
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.formulations.sequential_sampling_settings import (
    SequentialSampling_Settings,
)
from gemseo_umdo.formulations.surrogate_settings import Surrogate_Settings
from gemseo_umdo.formulations.taylor_polynomial_settings import (
    TaylorPolynomial_Settings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.discipline import Discipline


@pytest.fixture
def disciplines() -> list[AnalyticDiscipline]:
    """The coupled disciplines."""
    disc0 = AnalyticDiscipline(
        {"f": "x0+y1+y2+u", "c": "x0+y1+y2+2*u", "o": "x0+y1+y2+3*u"}, name="D0"
    )
    disc1 = AnalyticDiscipline({"y1": "x0+x1+2*y2+u1"}, name="D1")
    disc2 = AnalyticDiscipline({"y2": "x0+x2+y1+u2"}, name="D2")
    return [disc0, disc1, disc2]


@pytest.fixture
def mdf_discipline() -> DisciplineChain:
    """A monodisciplinary version of `disciplines`."""
    disc0 = AnalyticDiscipline(
        {"f": "x0+y1+y2+u", "c": "x0+y1+y2+2*u", "o": "x0+y1+y2+3*u"}, name="D0"
    )
    disc1 = AnalyticDiscipline({"y1": "-(3*x0+x1+2*x2+u1+2*u2)"}, name="D1")
    disc2 = AnalyticDiscipline({"y2": "-(2*x0+x1+x2+u1+u2)"}, name="D2")
    return DisciplineChain([disc1, disc2, disc0])


@pytest.fixture
def design_space() -> DesignSpace:
    """The design space."""
    space = DesignSpace()
    space.add_variable("x0", lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("x1", lower_bound=0.0, upper_bound=2.0, value=0.5)
    space.add_variable("x2", lower_bound=0.0, upper_bound=3.0, value=0.5)
    return space


@pytest.fixture
def uncertain_space() -> RandomSpace:
    """The uncertain space."""
    space = RandomSpace()
    space.add_variable("u", SPNormalDistribution_Settings(mu=1.0, sigma=1.0))
    space.add_variable("u1", SPNormalDistribution_Settings(mu=2.0, sigma=2.0))
    space.add_variable("u2", SPNormalDistribution_Settings(mu=3.0, sigma=3.0))
    return space


@pytest.fixture
def mdo_formulation(
    disciplines: Sequence[Discipline], uncertain_space: RandomSpace
) -> MDF:
    """The MDO formulation."""
    # TODO(bump-gemseo): pass the problem first, e.g. OptimizationProblem(design_space), then set its objective; the loose settings go into settings=<Formulation>_Settings(...)  # noqa: E501
    return MDF(disciplines, "f", uncertain_space)


@pytest.fixture
def quadratic_problem() -> tuple[AnalyticDiscipline, DesignSpace, RandomSpace]:
    """The discipline, design space and uncertain space of a quadratic problem."""
    discipline = AnalyticDiscipline(
        {"y": "(x+u)**2+(z+v)**3"}, name="quadratic_function"
    )

    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1, upper_bound=1.0, value=0.5)

    uncertain_space = RandomSpace()
    uncertain_space.add_variable("u", OTNormalDistribution_Settings())

    return discipline, design_space, uncertain_space


_SETTINGS = (
    Sampling_Settings(n_samples=10),
    Sampling_Settings(n_samples=10, estimate_statistics_iteratively=False),
    SequentialSampling_Settings(n_samples=10),
    SequentialSampling_Settings(n_samples=10, estimate_statistics_iteratively=False),
    TaylorPolynomial_Settings(),
    TaylorPolynomial_Settings(second_order=True),
)


@pytest.fixture(params=_SETTINGS)
def statistic_estimation_settings_for_dirac(request):
    """Statistic estimation settings compatible with the Dirac distribution."""
    return request.param


@pytest.fixture(
    params=[
        *_SETTINGS,
        ControlVariate_Settings(n_samples=10),
        PCE_Settings(n_samples=20),
        Surrogate_Settings(n_samples=20),
    ],
)
def statistic_estimation_settings(request):
    """Statistic estimation settings."""
    return request.param
