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
"""An uncertain coupling graph."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Final

from gemseo.core.dependency_graph import DependencyGraph
from gemseo.dataset.dataset import Dataset
from gemseo.post.graph_view import GraphView
from gemseo.scenario import EvaluationScenario
from gemseo.uncertainty.statistic.empirical import EmpiricalStatistics
from gemseo.util.discipline import get_all_outputs
from gemseo.util.string import pretty_repr
from gemseo.util.string import repr_variable
from numpy import absolute
from numpy import errstate
from numpy import hstack
from numpy import inf
from numpy import isfinite
from numpy import nan_to_num

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence
    from pathlib import Path

    from gemseo.discipline import Discipline
    from gemseo.doe.core.base_doe_settings import BaseDOESettings
    from gemseo.space.base import BaseVariableSpace
    from gemseo.util.typing import RealArray


class UncertainCouplingGraph:
    """An uncertain coupling graph.

    A coupling graph whose disciplines are represented by nodes
    and coupling variables by edges whose thickness is proportional to its dispersion.

    The dispersion is computed using a
    [DispersionMeasure][gemseo_umdo.visualization.uncertain_coupling_graph.UncertainCouplingGraph.DispersionMeasure]
    such as the coefficient of variation (CV)
    or the quartile coefficient of dispersion (QCD).

    To be used as:

    1. Instantiate an
       [UncertainCouplingGraph][gemseo_umdo.visualization.uncertain_coupling_graph.UncertainCouplingGraph].
    2. Sample the multidisciplinary system, using
       [sample()][gemseo_umdo.visualization.uncertain_coupling_graph.UncertainCouplingGraph.sample].
    3. Generate the coupling graph for a given dispersion measure, using
       [visualize()][gemseo_umdo.visualization.uncertain_coupling_graph.UncertainCouplingGraph.visualize].

    If you want to change the dispersion measure or filter the variables,
    repeat Step 3 with another dispersion measure or a list of variable names.

    If you want to improve the estimations of the statistics,
    repeat Step 2 with additional evaluations and Step 3.
    """

    class DispersionMeasure(StrEnum):
        """A dispersion measure."""

        CV = "CV"
        QCD = "QCD"

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        uncertain_space: BaseVariableSpace,
        variable_names: Iterable[str] | None = None,
    ) -> None:
        """
        Args:
            disciplines: The coupled disciplines.
            uncertain_space: The space of the uncertain variables.
            variable_names: The names of the coupling variables of interest.
                If `None`, use all the coupling variables.
        """  # noqa: D205 D212 D415
        if variable_names is None:
            self.__output_names = get_all_outputs(disciplines)
        else:
            self.__output_names = tuple(variable_names)

        self.__scenario = EvaluationScenario(disciplines, uncertain_space)
        for output_name in self.__output_names:
            self.__scenario.add_observable(output_name)

    def sample(self, algorithm_settings: BaseDOESettings) -> None:
        """Sample the multidisciplinary system.

        Args:
            algorithm_settings: The settings of the DOE algorithm,
                e.g. `OT_OPT_LHS_Settings(n_samples=100)`.
        """
        self.__scenario.execute(algorithm_settings)

    def visualize(
        self,
        maximum_thickness: int = 30,
        dispersion_measure: DispersionMeasure = DispersionMeasure.QCD,
        variable_names: Iterable[str] | None = None,
        show: bool = True,
        save: bool = True,
        file_path: str | Path = "",
        clean_up: bool = True,
    ) -> GraphView:
        """Generate the uncertain coupling graph.

        Args:
            maximum_thickness: The maximum thickness of a line,
                given to the edge of the most dispersed coupling.
            dispersion_measure: A standardized measure of dispersion.
            variable_names: The names of the coupling variables of interest,
                among those defined at instantiation.
                If `None`,
                use all the coupling variables of interest defined at instantiation.
            show: Whether to display the graph
                with the default application associated to the file extension.
            save: Whether to save the graph on the disk.
            file_path: The file path with extension to save the graph.
                If `""`, use the class name with PNG format.
            clean_up: Whether to remove the source files.

        Returns:
            The view of the uncertain coupling graph.

        Raises:
            ValueError: When a variable of interest
                was not defined as such at instantiation.

        Note:
            The thickness of an edge is proportional to the absolute dispersion
            of its coupling,
            the most dispersed coupling being drawn with the maximum thickness.
            An infinite dispersion,
            e.g. the coefficient of variation of a non-constant component
            whose mean is zero,
            is drawn with the maximum thickness,
            while an undefined dispersion,
            e.g. the coefficient of variation of a component equal to zero,
            is drawn with zero thickness.
        """
        if variable_names is None:
            variable_names = self.__output_names

        variable_names = set(variable_names)
        unobserved_names = variable_names.difference(self.__output_names)
        if unobserved_names:
            msg = (
                "The variables of interest must be among those defined "
                f"at instantiation; got unknown ones: {pretty_repr(unobserved_names)}."
            )
            raise ValueError(msg)

        dependency_graph = DependencyGraph(self.__scenario.disciplines).graph
        edges = [
            (head_disc, tail_disc, variable_names.intersection(coupling_names))
            for head_disc, tail_disc, coupling_names in dependency_graph.edges(
                data="io"
            )
        ]
        edges.extend(
            (
                discipline,
                discipline,
                variable_names.intersection(
                    discipline.io.input_grammar.names
                ).intersection(discipline.io.output_grammar.names),
            )
            for discipline in dependency_graph.nodes
        )
        graph_view = GraphView()
        for discipline in self.__scenario.disciplines:
            graph_view.node(discipline.name)

        coupling_names = set().union(*(names for _, _, names in edges))
        output_names_to_measures = {}
        # Skip an empty selection, for which there is no history to stack.
        if coupling_names:
            # EvaluationScenario.to_dataset() exports only the last DOE samples,
            # whereas the database of its problem accumulates those of all the runs;
            # only the histories of the coupling variables of interest are read.
            database = self.__scenario.formulation.problem.database
            name_to_history = {}
            for name in sorted(coupling_names):
                history = database.get_function_history(name)
                name_to_history[name] = history.reshape(len(history), -1)

            dataset = Dataset.from_array(
                hstack(list(name_to_history.values())),
                list(name_to_history),
                {name: history.shape[1] for name, history in name_to_history.items()},
            )
            with errstate(divide="ignore", invalid="ignore"):
                output_names_to_measures = _disp_meas_to_function[dispersion_measure](
                    EmpiricalStatistics(dataset)
                )

            # The absolute measures are scaled by the largest finite one,
            # so that the most dispersed coupling is drawn with the maximum thickness;
            # an infinite measure is also drawn with the maximum thickness
            # and an undefined one with zero thickness.
            output_names_to_measures = {
                name: nan_to_num(absolute(measure), nan=0.0, posinf=inf)
                for name, measure in output_names_to_measures.items()
            }
            largest_finite_measure = max(
                (
                    measure[isfinite(measure)].max(initial=0.0)
                    for measure in output_names_to_measures.values()
                ),
                default=0.0,
            )
            if largest_finite_measure == 0.0:
                largest_finite_measure = 1.0

            output_names_to_measures = {
                name: nan_to_num(measure / largest_finite_measure, posinf=1.0)
                for name, measure in output_names_to_measures.items()
            }

        for head_disc, tail_disc, names in edges:
            for coupling_name in sorted(names):
                disp_meas = output_names_to_measures[coupling_name]
                coupling_size = disp_meas.size
                for i in range(coupling_size):
                    graph_view.edge(
                        head_disc.name,
                        tail_disc.name,
                        label=repr_variable(coupling_name, i, coupling_size),
                        penwidth=str(round(disp_meas[i] * maximum_thickness, 2)),
                    )

        if save:
            graph_view.visualize(show=show, file_path=file_path, clean_up=clean_up)

        return graph_view


_disp_meas_to_function: Final[
    dict[
        UncertainCouplingGraph.DispersionMeasure,
        Callable[[EmpiricalStatistics], dict[str, RealArray]],
    ]
] = {
    UncertainCouplingGraph.DispersionMeasure.CV: (
        EmpiricalStatistics.compute_variation_coefficient
    ),
    UncertainCouplingGraph.DispersionMeasure.QCD: (
        EmpiricalStatistics.compute_quartile_coefficient_of_dispersion
    ),
}
"""The functions computing the dispersion measures from empirical statistics."""
