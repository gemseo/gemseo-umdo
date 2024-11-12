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

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Final

from gemseo.core.dependency_graph import DependencyGraph
from gemseo.disciplines.utils import get_all_outputs
from gemseo.post._graph_view import GraphView
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.utils.string_tools import repr_variable
from numpy import atleast_1d
from numpy import quantile
from strenum import StrEnum

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from pathlib import Path

    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline.discipline import Discipline
    from numpy.typing import NDArray


def _compute_qcd(x: NDArray[float]) -> NDArray[float]:
    """Compute the quartile coefficient of dispersion.

    Args:
        x: The data to compute the quartile coefficient of dispersion.

    Returns:
        The quartile coefficient of dispersion.
    """
    q025 = quantile(x, 0.25, 0)
    q075 = quantile(x, 0.75, 0)
    return (q075 - q025) / (q025 + q075)


class UncertainCouplingGraph:
    """An uncertain coupling graph.

    A coupling graph whose disciplines are represented by nodes
    and coupling variables by edges whose thickness is proportional to its dispersion.

    The dispersion is computed using a
    [DispersionMeasure][gemseo_umdo.visualizations.uncertain_coupling_graph.UncertainCouplingGraph.DispersionMeasure]
    such as the coefficient of variation (CV)
    or the quartile coefficient of dispersion (QCD).

    To be used as:

    1. Instantiate an
       [UncertainCouplingGraph][gemseo_umdo.visualizations.uncertain_coupling_graph.UncertainCouplingGraph].
    2. Sample the multidisciplinary system, using
       [sample()][gemseo_umdo.visualizations.uncertain_coupling_graph.UncertainCouplingGraph.sample].
    3. Generate the coupling graph for a given dispersion measure, using
       [visualize()][gemseo_umdo.visualizations.uncertain_coupling_graph.UncertainCouplingGraph.visualize].

    If you want to change the dispersion measure or filter the variables,
    repeat Step 3 with another dispersion measure or a list of variable names.

    If you want to improve the estimations of the statistics,
    repeat Step 2 with additional evaluations and Step 3.
    """

    class DispersionMeasure(StrEnum):
        """A dispersion measure."""

        CV = "CV"
        QCD = "QCD"

    __DISP_MEAS_TO_FUNCTION: Final[dict[DispersionMeasure, Callable]] = {
        DispersionMeasure.CV: lambda x: x.std(0) / x.mean(),
        DispersionMeasure.QCD: lambda x: _compute_qcd(x),
    }

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        uncertain_space: ParameterSpace,
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
            self.__output_names = variable_names

        self.__scenario = DOEScenario(
            disciplines, self.__output_names[0], uncertain_space, formulation_name="MDF"
        )
        for output_name in self.__output_names:
            self.__scenario.add_observable(output_name)

    def sample(
        self, n_samples: int, algo_name: str = "OT_OPT_LHS", **algo_options: Any
    ) -> None:
        """Sample the multidisciplinary system.

        Args:
            n_samples: The number of evaluations of the multidisciplinary system.
            algo_name: The name of the DOE algorithm.
            **algo_options: The options of the DOE algorithm.
        """
        self.__scenario.execute(
            algo_name=algo_name, n_samples=n_samples, **algo_options
        )

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
            maximum_thickness: The maximum thickness of a line.
            dispersion_measure: A standardized measure of dispersion.
            variable_names: The names of the coupling variables of interest.
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
        """
        if variable_names is None:
            all_output_names = self.__output_names
        else:
            all_output_names = variable_names

        database = self.__scenario.formulation.optimization_problem.database
        output_names_to_measures = {
            output_name: self.__DISP_MEAS_TO_FUNCTION[dispersion_measure](
                database.get_function_history(output_name)
            )
            for output_name in self.__output_names
        }
        dependency_graph = DependencyGraph(self.__scenario.disciplines).graph
        graph_view = GraphView()
        for discipline in self.__scenario.disciplines:
            graph_view.node(discipline.name)

        for head_disc, tail_disc, coupling_names in dependency_graph.edges(data="io"):
            variable_names = set(coupling_names).intersection(set(all_output_names))
            for coupling_name in variable_names:
                disp_meas = atleast_1d(output_names_to_measures[coupling_name])
                coupling_size = disp_meas.size
                for i in range(coupling_size):
                    graph_view.edge(
                        head_disc.name,
                        tail_disc.name,
                        label=repr_variable(coupling_name, i, coupling_size),
                        penwidth=str(round(abs(disp_meas[i] * maximum_thickness), 2)),
                    )

        for discipline in dependency_graph.nodes:
            coupling_names = set(discipline.io.input_grammar.names).intersection(
                discipline.io.output_grammar.names
            )
            discipline_name = discipline.name
            variable_names = set(coupling_names).intersection(set(all_output_names))
            for coupling_name in variable_names:
                disp_meas = atleast_1d(output_names_to_measures[coupling_name])
                coupling_size = disp_meas.size
                for i in range(coupling_size):
                    graph_view.edge(
                        discipline_name,
                        discipline_name,
                        label=repr_variable(coupling_name, i, coupling_size),
                        penwidth=str(round(abs(disp_meas[i] * maximum_thickness), 2)),
                    )

        if save:
            graph_view.visualize(show=show, file_path=file_path, clean_up=clean_up)

        return graph_view
