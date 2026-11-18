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

import contextlib
import re
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from gemseo.core.function.array_function import ArrayFunction
from gemseo.discipline import AnalyticDiscipline
from gemseo.discipline import AutoPyDiscipline
from gemseo.discipline import DisciplineChain
from gemseo.doe import CustomDOE_Settings
from gemseo.formulation import MDF_Settings
from gemseo.formulation.mdf import MDF
from gemseo.mda import MDAChain_Settings
from gemseo.mda import MDAGaussSeidel_Settings
from gemseo.optimization import NLOPT_COBYLA_Settings
from gemseo.optimization import SLSQP_Settings
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from gemseo.util.testing.helper import assert_exception
from numpy import array
from numpy import atleast_2d
from numpy import vstack
from numpy.linalg import norm
from numpy.testing import assert_equal

from gemseo_umdo.discipline.noiser.additive import AdditiveNoiser
from gemseo_umdo.discipline.noiser.multiplicative import MultiplicativeNoiser
from gemseo_umdo.formulation.factory import UMDOFormulationFactory
from gemseo_umdo.formulation.sampling import Sampling
from gemseo_umdo.formulation.sampling_settings import Sampling_Settings
from gemseo_umdo.scenario.mdo import UMDOScenario
from gemseo_umdo.statistic.settings.margin import Margin_Settings
from gemseo_umdo.statistic.settings.probability import Probability_Settings

if TYPE_CHECKING:
    from collections.abc import Generator

    from gemseo_umdo.formulation.core.base import BaseUMDOFormulation


@pytest.fixture
def scenario(toy_disciplines, toy_design_space, toy_uncertain_space) -> UMDOScenario:
    """The MDO scenario under uncertainty."""
    scn = UMDOScenario(
        toy_disciplines,
        toy_design_space,
        toy_uncertain_space,
        statistic_estimation_settings=Sampling_Settings(n_samples=3),
        formulation_settings=MDF_Settings(
            main_mda_settings=MDAChain_Settings(
                inner_mda_settings=MDAGaussSeidel_Settings()
            )
        ),
    )
    scn.add_objective("f", "Mean")
    scn.add_constraint("c", Margin_Settings(factor=3.0))
    scn.add_observable("o", "Mean")
    return scn


def test_disciplines_copy(scenario, toy_disciplines):
    """Check that the BaseUScenario works on a copy of the sequence of disciplines."""
    assert id(scenario.disciplines) != id(toy_disciplines)


def test_available_statistics(scenario):
    """Check the property returning the names of the available statistics."""
    expected = set(scenario.formulation.available_statistics)
    assert set(scenario.available_statistics) == expected


def test_factory(scenario):
    """Check that the UMDOScenario uses a UMDOFormulationFactory."""
    assert isinstance(scenario._formulation_factory, UMDOFormulationFactory)


def test_formulation(scenario):
    """Check the formulation after instantiation."""
    assert isinstance(scenario.formulation, Sampling)
    assert scenario.formulation_name == "Sampling[MDF; OT_OPT_LHS(3)]"
    assert scenario.formulation.doe_algo.algo_name == "OT_OPT_LHS"
    assert scenario.formulation._settings.doe_algo_settings.n_samples == 3


def test_design_space(scenario):
    """Check that the design space contains the design variables."""
    assert set(scenario.design_space.variables) == {"x0", "x1", "x2"}


def test_uncertain_space(scenario):
    """Check that the uncertain space contains the uncertain variables."""
    assert set(scenario.uncertain_space.variables) == {"u"}


def test_repr(scenario):
    """Check the string representation of the scenario."""
    expected = """UMDOScenario
   Disciplines: D0 D1 D2
   Formulation:
      MDO formulation: MDF
      Statistic estimation: Sampling
   Uncertain space:
      +------+-------------------------+
      | Name |       Distribution      |
      +------+-------------------------+
      | u    | norm(mu=0.0, sigma=1.0) |
      +------+-------------------------+"""
    assert repr(scenario) == expected


def test_mdo_formulation(scenario):
    """Check the content of the MDO formulation."""
    mdo_formulation = scenario.mdo_formulation
    opt_problem = mdo_formulation.problem
    assert isinstance(mdo_formulation, MDF)
    assert mdo_formulation.mda.inner_mdas[0].name == "MDAGaussSeidel"
    assert mdo_formulation.disciplines == scenario.disciplines
    assert [o.name for o in opt_problem.observables] == ["f", "c", "o"]
    assert list(scenario.mdo_formulation.input_space.variables) == ["u"]


def test_pydantic_mdo_formulation(
    toy_disciplines, toy_design_space, toy_uncertain_space
):
    """Check that the MDO formulation can be passed as a Pydantic model."""
    scenario = UMDOScenario(
        toy_disciplines,
        toy_design_space,
        toy_uncertain_space,
        statistic_estimation_settings=Sampling_Settings(n_samples=3),
        formulation_settings=MDF_Settings(),
    )
    scenario.add_objective("f", "Mean")
    assert isinstance(scenario.mdo_formulation, MDF)


@pytest.mark.parametrize(
    "minimize", [None, True, False], ids=["default", "minimize", "maximize"]
)
def test_minimize_objective(
    toy_disciplines, toy_design_space, toy_uncertain_space, minimize
):
    """Check that the argument minimize of add_objective is correctly used."""
    kwargs = {} if minimize is None else {"minimize": minimize}
    scn = UMDOScenario(
        toy_disciplines,
        toy_design_space,
        toy_uncertain_space,
        statistic_estimation_settings=Sampling_Settings(n_samples=3),
    )
    scn.add_objective("f", "Mean", **kwargs)
    maximize = minimize is False
    assert scn.formulation.problem.minimize_objective is not maximize
    expected_name = "-E[f]" if maximize else "E[f]"
    assert scn.formulation.problem.objective.name == expected_name


def test_uncertain_design_variables(
    toy_disciplines, toy_design_space, toy_uncertain_space
):
    """Check that a design variable can be noised.

    Here we check the disciplines created by the BaseUScenario.
    """
    scn = UMDOScenario(
        toy_disciplines,
        toy_design_space,
        toy_uncertain_space,
        statistic_estimation_settings=Sampling_Settings(n_samples=3),
        uncertain_design_variables={
            "x0": ("+", "v0"),
            "x1": "{}+v1",
            "x2": ("*", "v2"),
        },
    )
    scn.add_objective("f", "Mean")
    design_space = scn.design_space
    for i in range(3):
        assert f"x{i}" not in design_space
        assert f"dv_x{i}" in design_space

    assert len(scn.disciplines) == len(toy_disciplines) + 1
    mdo_chain = scn.disciplines[0]
    assert isinstance(mdo_chain, DisciplineChain)

    discipline = mdo_chain.disciplines[0]
    assert isinstance(discipline, AnalyticDiscipline)
    assert discipline.expressions == {"x1": "dv_x1+v1"}

    discipline = mdo_chain.disciplines[1]
    assert isinstance(discipline, AdditiveNoiser)
    assert set(discipline.input_grammar.names) == {"dv_x0", "v0"}
    assert set(discipline.output_grammar.names) == {"x0"}

    discipline = mdo_chain.disciplines[2]
    assert isinstance(discipline, MultiplicativeNoiser)
    assert set(discipline.input_grammar.names) == {"dv_x2", "v2"}
    assert set(discipline.output_grammar.names) == {"x2"}


@pytest.mark.parametrize("x", [array([1, 2]), array([1])])
@pytest.mark.parametrize(
    ("u1", "u2"),
    [(array([1.0]), array([-1.0])), (array([1.0, -1.0]), array([-1.0, 1.0]))],
)
def test_uncertain_design_variables_values(x, u1, u2):
    """Check that a design variable can be noised.

    Here we check the disciplines.
    """
    uncertain_space = RandomSpace()
    uncertain_space.add_variable("u", *[OTNormalDistribution_Settings()] * len(u1))

    def f(x):
        y = norm(x) ** 2
        return y  # noqa: RET504

    design_space = DesignSpace()
    design_space.add_variable("x", size=len(x), value=0)

    discipline = AutoPyDiscipline(f)
    scenario = UMDOScenario(
        [discipline],
        design_space,
        uncertain_space,
        statistic_estimation_settings=Sampling_Settings(
            doe_algo_settings=CustomDOE_Settings(samples=vstack((u1, u2)))
        ),
        uncertain_design_variables={"x": ("+", "u")},
    )
    scenario.add_objective("y", "Mean")
    scenario.execute(CustomDOE_Settings(samples=atleast_2d(x)))
    assert scenario.optimization_result.f_opt == (f(x + u1) + f(x + u2)) / 2


def test_statistic_no_estimation_parameters(
    toy_disciplines, toy_design_space, toy_uncertain_space
):
    """Check that a TypeError is raised when estimation settings are missing."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            "__init__() missing 1 required positional argument: "
            "'statistic_estimation_settings'"
        ),
    ):
        UMDOScenario(toy_disciplines, toy_design_space, toy_uncertain_space)


@pytest.mark.parametrize(
    ("constraint_name", "constraint_expr", "constraint_res"),
    [
        (None, "E[y] <= 0.05", "[E[y]-0.05] = [1.45]"),
        ("foo", "foo: E[y] <= 0.05", "foo = [1.45]"),
    ],
)
@pytest.mark.parametrize(
    ("maximize_objective", "use_standardized_objective", "objective_expr"),
    [
        (False, False, "minimize E[y]"),
        (False, True, "minimize E[y]"),
        (True, False, "maximize E[y]"),
        (True, True, "minimize -E[y]"),
    ],
)
def test_log(
    caplog,
    constraint_name,
    constraint_expr,
    constraint_res,
    maximize_objective,
    use_standardized_objective,
    objective_expr,
):
    """Check some parts of the log of a scenario."""
    discipline = AnalyticDiscipline({"y": "x**2+u"}, name="f")

    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1, upper_bound=1.0, value=0.5)

    uncertain_space = RandomSpace()
    uncertain_space.add_variable("u", OTNormalDistribution_Settings())

    scenario = UMDOScenario(
        [discipline],
        design_space,
        uncertain_space,
        statistic_estimation_settings=Sampling_Settings(
            doe_algo_settings=CustomDOE_Settings(samples=array([[0.5]]))
        ),
    )
    scenario.add_objective("y", "Mean", minimize=not maximize_objective)

    scenario.add_constraint(
        "y",
        "Mean",
        constraint_type="ineq",
        value=0.05,
        constraint_name=constraint_name,
    )
    scenario.use_standardized_objective = use_standardized_objective
    scenario.execute(CustomDOE_Settings(samples=array([[1.0]])))
    assert objective_expr in caplog.text
    assert constraint_expr in caplog.text
    assert constraint_res in caplog.text


@contextlib.contextmanager
def record_joint_sampling_flags(
    formulation: BaseUMDOFormulation,
) -> Generator[list[bool], None, None]:
    """Record whether the joint sampling is enabled at each sampling.

    Args:
        formulation: The U-MDO formulation.

    Yields:
        The values of `estimate_statistic_jacobians` at each sampling.
    """
    flags = []
    compute_samples = formulation.compute_samples

    def record_flag(*args, **kwargs):
        flags.append(formulation.estimate_statistic_jacobians)
        return compute_samples(*args, **kwargs)

    with mock.patch.object(formulation, "compute_samples", side_effect=record_flag):
        yield flags


def create_toy_scenario(
    discipline: AnalyticDiscipline,
    design_space: DesignSpace,
    uncertain_space: RandomSpace,
) -> UMDOScenario:
    """Create a U-MDO scenario without objective from the toy problem.

    Args:
        discipline: The discipline.
        design_space: The design space.
        uncertain_space: The uncertain space.

    Returns:
        The U-MDO scenario.
    """
    return UMDOScenario(
        [discipline], design_space, uncertain_space, Sampling_Settings(n_samples=3)
    )


@pytest.mark.parametrize(
    ("algorithm_settings", "differentiation_method", "expected"),
    [
        pytest.param(SLSQP_Settings(max_iter=2), "user", True, id="SLSQP-user"),
        pytest.param(
            SLSQP_Settings(max_iter=2),
            "finite_differences",
            False,
            id="SLSQP-finite_differences",
        ),
        pytest.param(NLOPT_COBYLA_Settings(max_iter=2), "user", False, id="COBYLA"),
        pytest.param(
            CustomDOE_Settings(samples=array([[0.5] * 3])), "user", False, id="DOE"
        ),
        pytest.param(
            CustomDOE_Settings(samples=array([[0.5] * 3]), eval_jac=True),
            "user",
            True,
            id="DOE-eval_jac",
        ),
    ],
)
def test_estimate_statistic_jacobians_from_driver(
    scenario,
    toy_disciplines,
    algorithm_settings,
    differentiation_method,
    expected,
    enable_discipline_statistics,
):
    """Check that the U-scenario sets estimate_statistic_jacobians from the driver.

    The disciplines must be linearized only when the driver uses the gradients
    of the statistics computed from the Jacobians of the disciplines,
    and not when it approximates them, e.g. with finite differences.
    """
    scenario.set_differentiation_method(differentiation_method)
    formulation = scenario.formulation
    formulation.estimate_statistic_jacobians = not expected
    with record_joint_sampling_flags(formulation) as flags:
        scenario.execute(algorithm_settings)

    assert flags[0] is expected
    n_linearizations = toy_disciplines[0].execution_statistics.n_linearizations
    assert (n_linearizations > 0) is expected


def add_probability_objective(scenario: UMDOScenario) -> None:
    """Add a probability to minimize as objective.

    Args:
        scenario: The U-MDO scenario.
    """
    scenario.add_objective("y", Probability_Settings(threshold=0.5))


def add_probability_objective_to_maximize(scenario: UMDOScenario) -> None:
    """Add a probability to maximize as objective.

    Args:
        scenario: The U-MDO scenario.
    """
    scenario.add_objective("y", Probability_Settings(threshold=0.5), minimize=False)


def add_probability_constraint(scenario: UMDOScenario) -> None:
    """Add a mean as objective and a positive probability with an offset as constraint.

    Args:
        scenario: The U-MDO scenario.
    """
    scenario.add_objective("z", "Mean")
    scenario.add_constraint(
        "y", Probability_Settings(threshold=0.5), positive=True, value=0.1
    )


def add_probability_observable(scenario: UMDOScenario) -> None:
    """Add a mean as objective and a probability as observable.

    Args:
        scenario: The U-MDO scenario.
    """
    scenario.add_objective("z", "Mean")
    scenario.add_observable("y", Probability_Settings(threshold=0.5))


@pytest.mark.parametrize(
    ("add_functions", "algorithm_settings"),
    [
        pytest.param(
            add_probability_objective, SLSQP_Settings(max_iter=2), id="objective"
        ),
        pytest.param(
            add_probability_objective_to_maximize,
            SLSQP_Settings(max_iter=2),
            id="objective-maximize",
        ),
        pytest.param(
            add_probability_constraint, SLSQP_Settings(max_iter=2), id="constraint"
        ),
        pytest.param(
            add_probability_observable,
            NLOPT_COBYLA_Settings(max_iter=2, evaluate_observable_jacobian=True),
            id="observable",
        ),
        pytest.param(
            add_probability_observable,
            CustomDOE_Settings(samples=array([[0.5]]), eval_jac=True),
            id="observable-DOE",
        ),
    ],
)
def test_unsupported_jacobian_from_driver(
    toy_problem,
    add_functions,
    algorithm_settings,
    snapshot,
    enable_discipline_statistics,
):
    """Check that a driver requiring an unsupported Jacobian raises an error.

    The error must be raised before any sampling.
    """
    scenario = create_toy_scenario(*toy_problem)
    add_functions(scenario)
    with assert_exception(NotImplementedError, snapshot):
        scenario.execute(algorithm_settings)

    assert not toy_problem[0].execution_statistics.n_executions


@pytest.mark.parametrize(
    "algorithm_settings",
    [
        pytest.param(NLOPT_COBYLA_Settings(max_iter=2), id="COBYLA"),
        pytest.param(SLSQP_Settings(max_iter=2), id="SLSQP"),
        pytest.param(CustomDOE_Settings(samples=array([[0.5]])), id="DOE"),
    ],
)
def test_unsupported_jacobian_not_required(toy_problem, algorithm_settings):
    """Check that a statistic whose Jacobian cannot be estimated can be used.

    The driver must not differentiate this statistic,
    e.g. an observable of an optimizer that does not evaluate its Jacobian.
    """
    scenario = create_toy_scenario(*toy_problem)
    add_probability_observable(scenario)
    scenario.execute(algorithm_settings)


def test_jacobian_of_non_statistic_observable_from_driver(toy_problem):
    """Check that a driver can differentiate an observable that is not a statistic.

    E.g. a function of the design variables added to the problem directly.
    """
    scenario = create_toy_scenario(*toy_problem)
    scenario.add_objective("z", "Mean")
    problem = scenario.formulation.problem
    problem.add_observable(
        ArrayFunction(lambda x: 2 * x, "g", jac=lambda x: array([[2.0]]))
    )
    scenario.execute(CustomDOE_Settings(samples=array([[0.5]]), eval_jac=True))
    assert_equal(problem.database.get_gradient_history("g"), array([[2.0]]))


def test_estimate_statistic_jacobians_without_new_iter_observable(toy_problem):
    """Check that an observable not evaluated at each new iteration is ignored.

    The driver evaluating the Jacobians of the observables
    evaluated at each new iteration
    will not request the Jacobian of this observable.
    """
    scenario = create_toy_scenario(*toy_problem)
    scenario.add_objective("z", Probability_Settings(threshold=0.5))
    scenario.add_observable("y", "Mean")
    problem = scenario.formulation.problem
    problem.new_iter_observables = []
    with record_joint_sampling_flags(scenario.formulation) as flags:
        scenario.execute(
            CustomDOE_Settings(
                samples=array([[0.5]]), evaluate_observable_jacobian=True
            )
        )

    assert not flags[0]


def test_optimizer_observable_jacobians(scenario):
    """Check that an optimizer evaluating the observable Jacobians samples once.

    A gradient-free optimizer evaluating the Jacobians of the observables
    must sample the disciplines once per design point,
    including the first one.
    """
    formulation = scenario.formulation
    with mock.patch.object(
        formulation, "compute_samples", wraps=formulation.compute_samples
    ) as compute_samples:
        scenario.execute(
            NLOPT_COBYLA_Settings(max_iter=2, evaluate_observable_jacobian=True)
        )

    assert compute_samples.call_count == len(formulation.problem.database)


def test_optimizer_observable_jacobians_without_observable(
    toy_disciplines, toy_design_space, toy_uncertain_space, enable_discipline_statistics
):
    """Check that a gradient-free optimizer without observables does not linearize.

    Evaluating the Jacobians of the observables
    must not linearize the disciplines
    when there is no observable whose Jacobian can be estimated.
    """
    scenario = UMDOScenario(
        toy_disciplines,
        toy_design_space,
        toy_uncertain_space,
        statistic_estimation_settings=Sampling_Settings(n_samples=3),
        formulation_settings=MDF_Settings(),
    )
    scenario.add_objective("f", "Mean")
    scenario.execute(
        NLOPT_COBYLA_Settings(max_iter=2, evaluate_observable_jacobian=True)
    )
    assert not any(d.execution_statistics.n_linearizations for d in toy_disciplines)


def test_no_joint_sampling_after_execution(
    scenario, toy_disciplines, enable_discipline_statistics
):
    """Check that the evaluations after an execution do not linearize the disciplines.

    E.g. a post-processing evaluating the statistics at new design points.
    """
    scenario.execute(SLSQP_Settings(max_iter=2))
    n_linearizations = [
        d.execution_statistics.n_linearizations for d in toy_disciplines
    ]
    scenario.formulation.problem.objective.evaluate(array([0.1, 0.2, 0.3, 0.4]))
    assert [
        d.execution_statistics.n_linearizations for d in toy_disciplines
    ] == n_linearizations
