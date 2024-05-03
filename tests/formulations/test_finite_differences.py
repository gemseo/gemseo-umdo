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
"""Test functions for the use of finite-differences approximation."""

from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.hashable_ndarray import HashableNdarray
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.utils.derivatives.finite_differences import FirstOrderFD
from numpy import array
from numpy.testing import assert_allclose

from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario


@pytest.mark.parametrize(
    ("statistic_estimation", "statistic_estimation_parameters", "expected"),
    [
        (
            "Sampling",
            {
                "algo": "CustomDOE",
                "algo_options": {"samples": array([[1.0]])},
                "estimate_statistics_iteratively": False,
            },
            [8.0, 16.0],
        ),
        (
            "Sampling",
            {
                "algo": "CustomDOE",
                "algo_options": {"samples": array([[1.0]])},
                "estimate_statistics_iteratively": True,
            },
            [8.0, 16.0],
        ),
        ("TaylorPolynomial", {"second_order": False}, [6.0, 12.0]),
        ("TaylorPolynomial", {"second_order": True}, [6.0, 12.0]),
    ],
)
def test_finite_differences(
    statistic_estimation, statistic_estimation_parameters, expected
):
    """Check that the finite-difference approximation of derivatives are ok."""
    discipline = AnalyticDiscipline({
        "f": "(x1+2*x2+u)**2",
        "c": "(x1+2*x2+u)**2",
        "o": "(x1+2*x2+u)**2",
    })

    design_space = DesignSpace()
    design_space.add_variable("x1", l_b=-1, u_b=1.0, value=0.5)
    design_space.add_variable("x2", l_b=-1, u_b=1.0, value=0.5)

    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("u", "OTNormalDistribution")

    scenario = UDOEScenario(
        [discipline],
        "DisciplinaryOpt",
        "f",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation=statistic_estimation,
        statistic_estimation_parameters=statistic_estimation_parameters,
    )
    scenario.add_constraint("c", "Mean")
    scenario.add_observable("o", "Mean")
    scenario.set_differentiation_method("finite_differences")
    scenario.execute({
        "algo": "CustomDOE",
        "algo_options": {"samples": array([[1.0, 1.0]]), "eval_jac": True},
    })
    # The database storing the samples is cleared after each sampling.
    assert not scenario.mdo_formulation.optimization_problem.database

    # input_data_to_output_samples stores the samples
    # at points (x1, x2), (x1+dx1, x2) and (x1, x2+dx2)
    # where (x1, x2) is the design value at the current iteration.
    input_data_to_output_data = scenario.formulation.input_data_to_output_data
    assert len(input_data_to_output_data) == 3
    step = FirstOrderFD._DEFAULT_STEP
    for key, expected_key in zip(
        input_data_to_output_data,
        [
            HashableNdarray(array([1.0, 1.0])),
            HashableNdarray(array([1.0 - step, 1.0])),
            HashableNdarray(array([1.0, 1.0 - step])),
        ],
    ):
        assert key == expected_key

    get_history = (
        scenario.formulation.optimization_problem.database.get_gradient_history
    )

    grad_history = get_history("E[f]")
    assert_allclose(grad_history, array([[expected]]), atol=1e-3)

    grad_history = get_history("E[o]")
    assert_allclose(grad_history, array([[expected]]), atol=1e-3)

    grad_history = get_history("E[c]")
    assert_allclose(grad_history, array([[expected]]), atol=1e-3)
