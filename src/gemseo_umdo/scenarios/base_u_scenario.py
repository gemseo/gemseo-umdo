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
"""Scenarios to address multidisciplinary design problems under uncertainty."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from gemseo.core.chains.chain import MDOChain
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.factory import MDOFormulationFactory
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

from gemseo_umdo.disciplines.noiser_factory import NoiserFactory
from gemseo_umdo.formulations.factory import UMDOFormulationsFactory

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.formulations.base_formulation_settings import BaseFormulationSettings
    from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
    from gemseo.typing import StrKeyMapping

    from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation


class BaseUScenario:
    """Base scenario for multidisciplinary design problems under uncertainty."""

    __DV_TAG: Final[str] = "{}"
    __DV_PREFIX: Final[str] = "dv_"
    formulation: BaseUMDOFormulation

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        objective_statistic_parameters: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        statistic_estimation: str = "Sampling",
        statistic_estimation_parameters: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        uncertain_design_variables: Mapping[
            str, str | tuple[str, str]
        ] = READ_ONLY_EMPTY_DICT,
        name: str = "",
        formulation_settings_model: BaseFormulationSettings | None = None,
        maximize_objective: bool = False,
        **formulation_settings: Any,
    ) -> None:
        """
        Args:
            uncertain_space: The uncertain variables
                with their probability distributions.
            objective_statistic_name: The name of the statistic
                to be applied to the objective, e.g. "margin".
            objective_statistic_parameters: The parameters of the statistics
                to be applied to the objective,
                e.g. `{"factor": 2.}` when `objective_statistic="margin"`.
            statistic_estimation: The name of the method to estimate the statistic.
            statistic_estimation_parameters: The options of `statistic_estimation`.
            uncertain_design_variables: This argument facilitates
                the definition of uncertain design variables in two ways.
                The first way consists of passing a dictionary
                of the form `{"x1": ("+", "u1"), "x2": ("*", "u2"), ...}`
                which defines the uncertain design variables as
                `x1 = dv_x1 + u1` and `x2 = dv_x2 * (1 + u2)`.
                Here `"x1"` and `"x2"` are the names of the design variables
                made uncertain by the random variables `"u_1"` and `"u_2"`
                which typically have zero mean.
                `"x1"` and `"x2"` are the names of the design variables
                actually used in `disciplines`
                while the names `"dv_x1"` and `dv_x2` are generated by the scenario.
                More generally,
                the first element of the tuple is assumed
                to be either the class name or the
                [SHORT_NAME][gemseo_umdo.disciplines.base_noiser.BaseNoiser.SHORT_NAME]
                of a
                [BaseNoiser][gemseo_umdo.disciplines.base_noiser.BaseNoiser]
                (feel free to create new noising disciplines).
                The second way of defining these uncertain design variables
                consists of passing a set of more complex expressions
                of the form `{"x": "{} + u", ...}`
                where `"x"` is the name of the design variable
                actually used in the equations,
                `"u"` is the name of the uncertain variable
                defined in the `uncertain_space`
                and `"{}"` is the optimization variable.
                Leave `"{}"` as is; it will be automatically replaced by `"dv_x"`.
                This more complex format assumes variables of dimension 1.
                If `None`,
                do not consider other variable relations
                than those defined by `disciplines`.
            maximize_objective: Whether to maximize the statistic of the objective.
        """  # noqa: D205 D212 D415
        disciplines = list(disciplines)
        if uncertain_design_variables:
            self.__add_noising_discipline_chain(
                disciplines, design_space, uncertain_design_variables
            )

        formulation_name = formulation_settings.pop("formulation_name")
        mdo_formulation_class = MDOFormulationFactory().get_class(formulation_name)

        # Create the design space associated with the optimization problem
        # generated by the MDO formulation
        mdo_formulation_design_space = mdo_formulation_class(
            disciplines,
            objective_name,
            design_space,
            settings_model=formulation_settings_model,
            **formulation_settings,
        ).design_space

        # Create the MDO formulation
        # whose functions are evaluable over the uncertain space
        # and differentiable with respect to the design variables.
        mdo_formulation = mdo_formulation_class(
            disciplines,
            objective_name,
            uncertain_space,
            differentiated_input_names_substitute=mdo_formulation_design_space.variable_names,  # noqa:E501
            **formulation_settings,
        )

        super().__init__(
            disciplines,
            objective_name,
            mdo_formulation_design_space,
            name=name,
            formulation_name=statistic_estimation,
            mdo_formulation=mdo_formulation,
            objective_statistic_name=objective_statistic_name,
            objective_statistic_parameters=objective_statistic_parameters,
            uncertain_space=uncertain_space,
            maximize_objective=maximize_objective,
            mdo_formulation_settings=formulation_settings,
            **statistic_estimation_parameters,
        )

        self.formulation_name = self.formulation.name

    def __add_noising_discipline_chain(
        self,
        disciplines: list[Discipline],
        design_space: DesignSpace,
        uncertain_design_variables: Mapping[str, str | tuple[str, str]],
    ) -> None:
        """Add noising disciplines to the user disciplines.

        Args:
            disciplines: The user disciplines to be extended with an `MDOChain`
                composed of noising discipline.
            design_space: The design space
                whose uncertain design variables are renamed by this method.
            uncertain_design_variables: The argument facilitating
                the definition of uncertain design variables.
        """
        noising_disciplines = []

        expressions = {
            dv_name: v
            for dv_name, v in uncertain_design_variables.items()
            if isinstance(v, str)
        }
        for dv_name, expression in expressions.items():
            new_dv_name = self.__get_design_variable_name(dv_name)
            design_space.rename_variable(dv_name, new_dv_name)
            expressions[dv_name] = expression.replace(self.__DV_TAG, new_dv_name)

        noising_disciplines.append(AnalyticDiscipline(expressions))
        for dv_name, v in uncertain_design_variables.items():
            if dv_name in expressions:
                continue

            new_dv_name = self.__get_design_variable_name(dv_name)
            design_space.rename_variable(dv_name, new_dv_name)
            noising_disciplines.append(
                NoiserFactory().create(v[0], new_dv_name, dv_name, v[1])
            )

        disciplines.insert(0, MDOChain(noising_disciplines))

    def __get_design_variable_name(self, uncertain_design_variable_name: str) -> str:
        """Return the name of the design variable to be noised.

        Args:
            uncertain_design_variable_name: The name of the uncertain design variable.

        Returns:
            The name of the design variable to be noised.
        """
        return f"{self.__DV_PREFIX}{uncertain_design_variable_name}"

    @property
    def _formulation_factory(self) -> UMDOFormulationsFactory:
        return UMDOFormulationsFactory()

    def add_constraint(
        self,
        output_name: str | Sequence[str],
        statistic_name: str,
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.INEQ,
        constraint_name: str = "",
        value: float = 0,
        positive: bool = False,
        **statistic_parameters: Any,
    ) -> None:
        """
        Args:
            statistic_name: The name of the statistic
                to be applied to the constraint, e.g. "margin".
            statistic_parameters: The parameters of the statistics
                to be applied to the constraint,
                `{"factor": 2.}` when `objective_statistic="margin"`.
        """  # noqa: D205 D212 D415
        self.formulation.add_constraint(
            output_name,
            statistic_name,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
            **statistic_parameters,
        )

    def add_observable(
        self,
        output_names: Sequence[str],
        statistic_name: str,
        observable_name: str = "",
        discipline: Discipline | None = None,
        **statistic_parameters: Any,
    ) -> None:
        """
        Args:
            statistic_name: The name of the statistic
                to be applied to the constraint, e.g. "margin".
            statistic_parameters: The parameters of the statistics
                to be applied to the constraint,
                `{"factor": 2.}` when `objective_statistic="margin"`.
        """  # noqa: D205 D212 D415
        self.formulation.add_observable(
            output_names,
            statistic_name,
            observable_name=observable_name,
            discipline=discipline,
            **statistic_parameters,
        )

    def __repr__(self) -> str:
        msg = MultiLineString()
        msg.add(self.name)
        msg.indent()
        msg.add("Disciplines: {}", pretty_str(self.disciplines, delimiter=" "))
        msg.add("Formulation:")
        msg.indent()
        msg.add("MDO formulation: {}", self.mdo_formulation.__class__.__name__)
        msg.add("Statistic estimation: {}", self.formulation.__class__.__name__)
        msg.dedent()
        msg.add("Uncertain space:")
        msg.indent()
        for line in str(self.uncertain_space).split("\n")[1:]:
            msg.add(line)
        return str(msg)

    @property
    def uncertain_space(self) -> ParameterSpace:
        """The uncertain variable space."""
        return self.formulation.uncertain_space

    @property
    def mdo_formulation(self) -> BaseMDOFormulation:
        """The MDO formulation over the uncertain space."""
        return self.formulation.mdo_formulation

    @property
    def available_statistics(self) -> list[str]:
        """The names of the available statistics."""
        return self.formulation.available_statistics
