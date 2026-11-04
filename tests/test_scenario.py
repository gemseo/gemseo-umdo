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
import gc
import re
import weakref
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
from gemseo.optimization import OptimizationProblem
from gemseo.optimization import SLSQP_Settings
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from gemseo.uncertainty.distribution import SPNormalDistribution_Settings
from gemseo.util.testing.helper import assert_exception
from numpy import array
from numpy import atleast_2d
from numpy import vstack
from numpy.linalg import norm
from numpy.testing import assert_almost_equal

from gemseo_umdo.disciplines.additive_noiser import AdditiveNoiser
from gemseo_umdo.disciplines.multiplicative_noiser import MultiplicativeNoiser
from gemseo_umdo.formulations._functions.base_statistic_function import (
    BaseStatisticFunction,
)
from gemseo_umdo.formulations.factory import UMDOFormulationsFactory
from gemseo_umdo.formulations.sampling import Sampling
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.u_evaluation_scenario import UEvaluationScenario
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator

    from gemseo.discipline import Discipline

    from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation


@pytest.fixture
def disciplines() -> list[Discipline]:
    """Three simple disciplines."""
    disc0 = AnalyticDiscipline(
        {"f": "x0+y1+y2", "c": "x0+y1+y2", "o": "x0+y1+y2"}, name="D0"
    )
    disc1 = AnalyticDiscipline({"y1": "x0+x1+2*y2"}, name="D1")
    disc2 = AnalyticDiscipline({"y2": "x0+x2+y1+u"}, name="D2")
    return [disc0, disc1, disc2]


@pytest.fixture
def design_space() -> DesignSpace:
    """The space of local and global design variables."""
    space = DesignSpace()
    for name in ["x0", "x1", "x2"]:
        space.add_variable(name, lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("y1", lower_bound=0.0, upper_bound=1.0, value=0.5)
    return space


@pytest.fixture
def uncertain_space() -> RandomSpace:
    """The space defining the uncertain variable."""
    space = RandomSpace()
    space.add_variable("u", SPNormalDistribution_Settings())
    return space


@pytest.fixture
def scenario(disciplines, design_space, uncertain_space) -> UMDOScenario:
    """The MDO scenario under uncertainty."""
    scn = UMDOScenario(
        disciplines,
        design_space,
        uncertain_space,
        statistic_estimation_settings=Sampling_Settings(n_samples=3),
        formulation_settings=MDF_Settings(
            main_mda_settings=MDAChain_Settings(
                inner_mda_settings=MDAGaussSeidel_Settings()
            )
        ),
    )
    scn.add_objective("f", "Mean")
    scn.add_constraint("c", "Margin", factor=3.0)
    scn.add_observable("o", "Mean")
    return scn


def test_disciplines_copy(scenario, disciplines):
    """Check that the BaseUScenario works on a copy of the sequence of disciplines."""
    assert id(scenario.disciplines) != id(disciplines)


def test_available_statistics(scenario):
    """Check the property returning the names of the available statistics."""
    expected = set(scenario.formulation.available_statistics)
    assert set(scenario.available_statistics) == expected


def test_factory(scenario):
    """Check that the UMDOScenario uses a UMDOFormulationsFactory."""
    assert isinstance(scenario._formulation_factory, UMDOFormulationsFactory)


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


def test_pydantic_mdo_formulation(disciplines, design_space, uncertain_space):
    """Check that the MDO formulation can be passed as a Pydantic model."""
    scenario = UMDOScenario(
        disciplines,
        design_space,
        uncertain_space,
        statistic_estimation_settings=Sampling_Settings(n_samples=3),
        formulation_settings=MDF_Settings(),
    )
    scenario.add_objective("f", "Mean")
    assert isinstance(scenario.mdo_formulation, MDF)


@pytest.mark.parametrize(
    "minimize", [None, True, False], ids=["default", "minimize", "maximize"]
)
def test_minimize_objective(disciplines, design_space, uncertain_space, minimize):
    """Check that the argument minimize of add_objective is correctly used."""
    kwargs = {} if minimize is None else {"minimize": minimize}
    scn = UMDOScenario(
        disciplines,
        design_space,
        uncertain_space,
        statistic_estimation_settings=Sampling_Settings(n_samples=3),
    )
    scn.add_objective("f", "Mean", **kwargs)
    maximize = minimize is False
    assert scn.formulation.problem.minimize_objective is not maximize
    expected_name = "-E[f]" if maximize else "E[f]"
    assert scn.formulation.problem.objective.name == expected_name


def test_uncertain_design_variables(disciplines, design_space, uncertain_space):
    """Check that a design variable can be noised.

    Here we check the disciplines created by the BaseUScenario.
    """
    scn = UMDOScenario(
        disciplines,
        design_space,
        uncertain_space,
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

    assert len(scn.disciplines) == len(disciplines) + 1
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


def test_statistic_no_estimation_parameters(disciplines, design_space, uncertain_space):
    """Check that a TypeError is raised when estimation settings are missing."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            "__init__() missing 1 required positional argument: "
            "'statistic_estimation_settings'"
        ),
    ):
        UMDOScenario(disciplines, design_space, uncertain_space)


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


@pytest.fixture
def u_evaluation_scenario(
    disciplines, design_space, uncertain_space
) -> UEvaluationScenario:
    """A UEvaluationScenario with an uncertain design variable."""
    return UEvaluationScenario(
        disciplines,
        design_space,
        uncertain_space,
        Sampling_Settings(n_samples=3),
        uncertain_design_variables={"x0": ("+", "u")},
    )


@pytest.mark.parametrize("method_name", ["add_objective", "add_constraint"])
def test_u_evaluation_scenario_no_objective_nor_constraint(
    u_evaluation_scenario, method_name
):
    """Check that a UEvaluationScenario cannot define an objective or constraints."""
    assert not hasattr(u_evaluation_scenario, method_name)


def test_u_evaluation_scenario_problem(u_evaluation_scenario):
    """Check that a UEvaluationScenario does not define an optimization problem."""
    assert not isinstance(
        u_evaluation_scenario.formulation.problem, OptimizationProblem
    )


def test_u_evaluation_scenario_formulation(u_evaluation_scenario):
    """Check that a UEvaluationScenario uses the U-MDO formulation of its settings."""
    assert isinstance(u_evaluation_scenario.formulation, Sampling)


def test_u_evaluation_scenario_repr(u_evaluation_scenario):
    """Check the string representation of a UEvaluationScenario."""
    assert repr(u_evaluation_scenario).startswith("UEvaluationScenario\n")


def test_u_evaluation_scenario_input_space(u_evaluation_scenario):
    """Check that the input space of a UEvaluationScenario includes the noising."""
    assert set(u_evaluation_scenario.input_space.variables) == {"dv_x0", "x1", "x2"}


def test_u_evaluation_scenario_observables(u_evaluation_scenario):
    """Check that a UEvaluationScenario evaluates statistics as observables."""
    u_evaluation_scenario.add_observable("o", "Mean")
    u_evaluation_scenario.add_observable("o", "Margin", factor=2.0)
    u_evaluation_scenario.execute(
        CustomDOE_Settings(samples=array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    )
    dataset = u_evaluation_scenario.to_dataset()
    assert set(dataset.output_names) == {"E[o]", "Margin[o; 2.0]"}
    assert len(dataset) == 2


def test_u_evaluation_scenario_values(toy_problem):
    """Check the statistics estimated by a UEvaluationScenario.

    The mean of $(x+u)^2$ over the samples $u=-1$ and $u=1$
    is the expectation $x^2+1$ of $(x+U)^2$ where $U$ is a standard Gaussian variable.
    """
    scenario = create_toy_u_evaluation_scenario(
        *toy_problem,
        Sampling_Settings(
            doe_algo_settings=CustomDOE_Settings(samples=array([[-1.0], [1.0]]))
        ),
    )
    scenario.execute(CustomDOE_Settings(samples=array([[-0.5], [0.0], [0.5]])))
    mean = scenario.to_dataset().get_view(variable_names="E[y]").to_numpy()
    assert_almost_equal(mean, array([[1.25], [1.0], [1.25]]))


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


@pytest.fixture
def toy_problem() -> tuple[AnalyticDiscipline, DesignSpace, RandomSpace]:
    """A discipline computing y=(x+u)**2 and z=x, its design and uncertain spaces."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1.0, upper_bound=1.0, value=0.5)
    uncertain_space = RandomSpace()
    uncertain_space.add_variable("u", SPNormalDistribution_Settings())
    discipline = AnalyticDiscipline({"y": "(x+u)**2", "z": "x"})
    return discipline, design_space, uncertain_space


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


def create_toy_u_evaluation_scenario(
    discipline: AnalyticDiscipline,
    design_space: DesignSpace,
    uncertain_space: RandomSpace,
    statistic_estimation_settings: Sampling_Settings,
) -> UEvaluationScenario:
    """Create a U-evaluation scenario observing the mean of y from the toy problem.

    Args:
        discipline: The discipline.
        design_space: The design space.
        uncertain_space: The uncertain space.
        statistic_estimation_settings: The statistic estimation settings.

    Returns:
        The U-evaluation scenario.
    """
    scenario = UEvaluationScenario(
        [discipline], design_space, uncertain_space, statistic_estimation_settings
    )
    scenario.add_observable("y", "Mean")
    return scenario


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
    disciplines,
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
    n_linearizations = disciplines[0].execution_statistics.n_linearizations
    assert (n_linearizations > 0) is expected


@pytest.mark.parametrize(
    ("algorithm_settings", "minimize", "add_as_constraint"),
    [
        pytest.param(SLSQP_Settings(max_iter=2), True, False, id="SLSQP"),
        pytest.param(SLSQP_Settings(max_iter=2), False, False, id="SLSQP-maximize"),
        pytest.param(SLSQP_Settings(max_iter=2), True, True, id="SLSQP-constraint"),
        pytest.param(
            CustomDOE_Settings(samples=array([[0.5]]), eval_jac=True),
            True,
            False,
            id="DOE-eval_jac",
        ),
    ],
)
def test_unsupported_jacobian_from_driver(
    toy_problem,
    algorithm_settings,
    minimize,
    add_as_constraint,
    enable_discipline_statistics,
    snapshot,
):
    """Check that a driver requiring an unsupported Jacobian does not linearize.

    The disciplines must not be linearized
    when the driver differentiates only statistics
    whose Jacobians cannot be estimated,
    even when their functions are transformed,
    e.g. the negation of an objective to maximize
    or of a positive constraint with an offset.
    """
    scenario = create_toy_scenario(*toy_problem)
    discipline = scenario.disciplines[0]
    if add_as_constraint:
        scenario.add_objective("z", "Probability", threshold=0.5)
        scenario.add_constraint(
            "y", "Probability", value=0.1, positive=True, threshold=0.5
        )
    else:
        scenario.add_objective("y", "Probability", minimize=minimize, threshold=0.5)

    with assert_exception(NotImplementedError, snapshot):
        scenario.execute(algorithm_settings)

    assert discipline.execution_statistics.n_linearizations == 0


def add_objective_to_maximize(scenario: UMDOScenario) -> None:
    """Add an objective to maximize, i.e. the negation of a statistic.

    Args:
        scenario: The U-MDO scenario.
    """
    scenario.add_objective("y", "Mean", minimize=False)


def add_positive_constraint(scenario: UMDOScenario) -> None:
    """Add a positive constraint with an offset, i.e. a transformed statistic.

    Args:
        scenario: The U-MDO scenario.
    """
    scenario.add_objective("z", "Probability", threshold=0.5)
    scenario.add_constraint("y", "Mean", value=0.1, positive=True)


def add_objective_toggled_twice(scenario: UMDOScenario) -> None:
    """Add an objective, then maximize it and minimize it again.

    Its function is then the negation of the negation of a statistic.

    Args:
        scenario: The U-MDO scenario.
    """
    scenario.add_objective("y", "Mean")
    problem = scenario.formulation.problem
    problem.minimize_objective = False
    problem.minimize_objective = True


def create_unused_objective(scenario: UMDOScenario) -> None:
    """Create an objective whose Jacobian can be estimated but not set it.

    Args:
        scenario: The U-MDO scenario.
    """
    scenario.add_objective("z", "Probability", threshold=0.5)
    scenario.formulation.create_objective("y", "Mean")


def remove_constraint(scenario: UMDOScenario) -> None:
    """Add a constraint whose Jacobian can be estimated, then remove it.

    Args:
        scenario: The U-MDO scenario.
    """
    scenario.add_objective("z", "Probability", threshold=0.5)
    scenario.add_constraint("y", "Mean")
    scenario.formulation.problem.constraints.clear()


@pytest.mark.parametrize(
    ("add_functions", "expected"),
    [
        # A transformation hides the statistic function,
        # whose Jacobian can be estimated.
        pytest.param(add_objective_to_maximize, True, id="objective-maximize"),
        pytest.param(add_positive_constraint, True, id="constraint-positive"),
        pytest.param(add_objective_toggled_twice, True, id="objective-toggled"),
        # The Jacobian of the statistic can be estimated
        # but the driver will not request it.
        pytest.param(create_unused_objective, False, id="unused-objective"),
        pytest.param(remove_constraint, False, id="removed-constraint"),
    ],
)
def test_estimate_statistic_jacobians_from_problem_functions(
    toy_problem,
    add_functions: Callable[[UMDOScenario], None],
    expected: bool,
):
    """Check that the joint sampling depends on the functions of the problem.

    Only the statistic functions of the objective and of the constraints
    of the problem, possibly transformed, must be considered.
    """
    scenario = create_toy_scenario(*toy_problem)
    add_functions(scenario)
    with (
        record_joint_sampling_flags(scenario.formulation) as flags,
        # The Jacobian of the objective "Probability" cannot be estimated.
        contextlib.suppress(NotImplementedError),
    ):
        scenario.execute(SLSQP_Settings(max_iter=2))

    assert flags[0] is expected


def test_unused_constraint_not_referenced(toy_problem):
    """Check that the formulation does not reference an unused constraint.

    A constraint created but not added to the problem
    must be garbage collected with its statistic function.
    """
    scenario = create_toy_scenario(*toy_problem)
    constraint = scenario.formulation.create_constraint("y", "Mean", positive=True)
    statistic_function = weakref.ref(
        BaseStatisticFunction.get_statistic_function(constraint)
    )
    del constraint
    gc.collect()
    assert statistic_function() is None


@pytest.mark.parametrize("negate", [False, True], ids=["function", "negation"])
def test_get_statistic_function_without_statistic_function(negate):
    """Check that a function without statistic function has none.

    The negation of a function is a function of its bound method,
    which is followed back to this function.
    """
    function = ArrayFunction(lambda x: x, "f")
    if negate:
        function = -function

    assert BaseStatisticFunction.get_statistic_function(function) is None


def test_objective_and_constraint_functions_without_objective(toy_problem):
    """Check the statistic functions of a problem without objective."""
    scenario = create_toy_scenario(*toy_problem)
    scenario.add_constraint("y", "Mean")
    formulation = scenario.formulation
    assert formulation._objective_and_constraint_functions == [
        BaseStatisticFunction.get_statistic_function(formulation.problem.constraints[0])
    ]


def test_estimate_statistic_jacobians_without_new_iter_observable(toy_problem):
    """Check that an observable not evaluated at each new iteration is ignored.

    The driver evaluating the Jacobians of the observables
    evaluated at each new iteration
    will not request the Jacobian of this observable.
    """
    scenario = create_toy_scenario(*toy_problem)
    scenario.add_objective("z", "Probability", threshold=0.5)
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
    disciplines, design_space, uncertain_space, enable_discipline_statistics
):
    """Check that a gradient-free optimizer without observables does not linearize.

    Evaluating the Jacobians of the observables
    must not linearize the disciplines
    when there is no observable whose Jacobian can be estimated.
    """
    scenario = UMDOScenario(
        disciplines,
        design_space,
        uncertain_space,
        statistic_estimation_settings=Sampling_Settings(n_samples=3),
        formulation_settings=MDF_Settings(),
    )
    scenario.add_objective("f", "Mean")
    scenario.execute(
        NLOPT_COBYLA_Settings(max_iter=2, evaluate_observable_jacobian=True)
    )
    assert not any(d.execution_statistics.n_linearizations for d in disciplines)


def test_no_joint_sampling_after_execution(
    scenario, disciplines, enable_discipline_statistics
):
    """Check that the evaluations after an execution do not linearize the disciplines.

    E.g. a post-processing evaluating the statistics at new design points.
    """
    scenario.execute(SLSQP_Settings(max_iter=2))
    n_linearizations = [d.execution_statistics.n_linearizations for d in disciplines]
    scenario.formulation.problem.objective.evaluate(array([0.1, 0.2, 0.3, 0.4]))
    assert [
        d.execution_statistics.n_linearizations for d in disciplines
    ] == n_linearizations


@pytest.mark.parametrize(
    ("eval_jac", "evaluate_observable_jacobian"),
    [
        pytest.param(False, True, id="evaluate_observable_jacobian"),
        # The evaluation problem has neither objective nor constraints.
        pytest.param(True, False, id="eval_jac"),
    ],
)
def test_u_evaluation_scenario_observable_jacobians(
    toy_problem, eval_jac, evaluate_observable_jacobian, enable_discipline_statistics
):
    """Check that a DOE evaluating the observable Jacobians samples once per point."""
    discipline = toy_problem[0]
    scenario = create_toy_u_evaluation_scenario(
        *toy_problem, Sampling_Settings(n_samples=5)
    )
    scenario.execute(
        CustomDOE_Settings(
            samples=array([[-0.5], [0.0], [0.5]]),
            eval_jac=eval_jac,
            evaluate_observable_jacobian=evaluate_observable_jacobian,
        )
    )
    statistics = discipline.execution_statistics
    assert statistics.n_executions == 15
    assert statistics.n_linearizations == 15
