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
"""A network of uncertain variables representing their Sobol' indices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.util.string import repr_variable
from graphviz import Digraph
from graphviz import nohtml
from numpy import atleast_1d
from numpy import atleast_2d

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.machine_learning.regression.model import PCERegressor
    from gemseo.uncertainty.sensitivity import SobolAnalysis
    from gemseo.util.typing import RealArray


class SobolGraph:
    """A network of uncertain variables representing their Sobol' indices.

    A node represents an uncertain variable whose name is written inside, followed by
    its total-order and first-order Sobol' indices in percent.

    The thickness of a node is proportional to the total-order Sobol' index of the
    variable while the thickness of an edge is proportional to the second-order Sobol'
    index of the corresponding pair of variables.

    To be used as:

    1. Instantiate a [SobolGraph][gemseo_umdo.visualization.sobol_graph.SobolGraph]
       from Sobol' indices,
       or from a Sobol' analysis with
       [from_analysis()][gemseo_umdo.visualization.sobol_graph.SobolGraph.from_analysis]
       or a PCE with
       [from_pce()][gemseo_umdo.visualization.sobol_graph.SobolGraph.from_pce].
    2. Draw the graph with
       [draw()][gemseo_umdo.visualization.sobol_graph.SobolGraph.draw],
       which returns a
       [graphviz.Digraph](https://graphviz.readthedocs.io/en/stable/api.html#graphviz.Digraph),
       displayed in a Jupyter notebook
       and saved to a file with
       [save_graph()][gemseo.post.graph.save_graph],
       e.g. `save_graph(sobol_graph.draw(), "sobol_graph.pdf")`,
       or with its `render()` method.
    """

    __first_order_indices: Mapping[str, float]
    """The first-order Sobol' indices of the scalar inputs."""

    __second_order_indices: Mapping[tuple[str, str], float]
    """The second-order Sobol' indices of the scalar inputs."""

    __total_order_indices: Mapping[str, float]
    """The total-order Sobol' indices of the scalar inputs."""

    def __init__(
        self,
        first_order_indices: Mapping[str, float],
        total_order_indices: Mapping[str, float],
        second_order_indices: Mapping[tuple[str, str], float],
    ) -> None:
        """
        Args:
            first_order_indices: The first-order Sobol' indices of the scalar inputs,
                shaped as `{name: index}`.
            total_order_indices: The total-order Sobol' indices of the scalar inputs,
                shaped as `{name: index}`.
            second_order_indices:  The second-order Sobol' indices of the scalar inputs,
                shaped as `{(name, other_name): index}`.
        """  # noqa: D205 D212 D415
        self.__first_order_indices = first_order_indices
        self.__second_order_indices = second_order_indices
        self.__total_order_indices = total_order_indices

    def draw(self, threshold: float = 0.1, maximum_thickness: float = 10.0) -> Digraph:
        """Draw the Sobol' graph.

        The graph is neither saved nor shown;
        use [save_graph()][gemseo.post.graph.save_graph]
        or its `render()` method to save it.

        Args:
            threshold: The sensitivity threshold
                above which a second-order index is significant
                and the corresponding edge drawn.
            maximum_thickness: The maximum thickness of a line.

        Returns:
            The Sobol' graph,
            whose nodes are named `v<i>`, `i` being the position of the variable,
            and labeled after the variables.
        """
        digraph = Digraph(
            node_attr={"style": "filled", "fillcolor": "white"},
            edge_attr={"dir": "none"},
        )

        # Add the nodes representing both first- and total-order indices.
        # The nodes are not named after the variables,
        # as graphviz reads a colon in a node name of an edge as a port separator.
        names_to_node_names = {}
        for i, (name, total_order_index) in enumerate(
            self.__total_order_indices.items()
        ):
            first_order_index = self.__first_order_indices[name]
            node_name = names_to_node_names[name] = f"v{i}"
            digraph.node(
                node_name,
                label=nohtml(
                    f"{name}\n"
                    f"({round(total_order_index * 100)}, "
                    f"{round(first_order_index * 100)})"
                ),
                penwidth=str(total_order_index * maximum_thickness),
            )

        # Add the edges representing the second-order indices.
        for (name, other_name), index in self.__second_order_indices.items():
            if index >= threshold:
                digraph.edge(
                    names_to_node_names[name],
                    names_to_node_names[other_name],
                    penwidth=str(index * maximum_thickness),
                )

        return digraph

    @staticmethod
    def __preprocess(indices: dict[str, RealArray]) -> dict[str, float]:
        """Convert indices expressed as NumPy arrays to float numbers.

        Args:
            indices: The indices of the variables as NumPy arrays.

        Returns:
            The indices of the variables as float numbers.
        """
        new_indices = {}
        for name, index in indices.items():
            size = index.size
            for i, sub_index in enumerate(index):
                new_indices[repr_variable(name, i, size)] = max(sub_index, 0)

        return new_indices

    @classmethod
    def __preprocess_second_order(
        cls, indices: dict[str, dict[str, RealArray]]
    ) -> dict[tuple[str, str], float]:
        """Convert second-order indices expressed as NumPy arrays to float numbers.

        Args:
            indices: The second-order indices of the variables as NumPy arrays.

        Returns:
            The second-order indices of the variables as float numbers.
        """
        new_indices = {}
        edges = []
        for name_1, indices_1 in indices.items():
            for name_2, indices_12 in indices_1.items():
                size_1, size_2 = indices_12.shape
                for component_1, sub_indices_12 in enumerate(indices_12):
                    for component_2, index in enumerate(sub_indices_12):
                        edge = tuple(
                            sorted([
                                repr_variable(name_1, component_1, size_1),
                                repr_variable(name_2, component_2, size_2),
                            ])
                        )
                        if edge not in edges:
                            new_indices[edge] = max(index, 0)
                            edges.append(edge)

        return new_indices

    @staticmethod
    def __check_output_name(output_name: str, output_names: Iterable[str]) -> None:
        """Check that a name is an output name.

        Args:
            output_name: The name to check.
            output_names: The output names.

        Raises:
            ValueError: When the name is not an output name.
        """
        if output_name not in output_names:
            msg = f"The name {output_name!r} is not an output name."
            raise ValueError(msg)

    @staticmethod
    def __check_output_component(
        output_name: str, output_component: int, size: int
    ) -> None:
        """Check that a component is in the range of an output.

        Args:
            output_name: The name of the output.
            output_component: The component of the output.
            size: The size of the output.

        Raises:
            ValueError: When the component is out of the range of the output.
        """
        if not 0 <= output_component < size:
            msg = (
                f"The component {output_component} of the output {output_name!r} "
                f"is not in the range [0, {size - 1}]."
            )
            raise ValueError(msg)

    @classmethod
    def from_analysis(
        cls, analysis: SobolAnalysis, output_name: str, output_component: int = 0
    ) -> SobolGraph:
        """Create the Sobol' graph from a Sobol' analysis.

        Args:
            analysis: A Sobol' analysis.
            output_name: The name of the output.
            output_component: The component of the output.

        Returns:
            The Sobol' graph associated with this Sobol' analysis.

        Raises:
            ValueError: When the name is not an output name
                or when the component is out of the range of the output.
        """
        first_order_indices = analysis.indices.first
        cls.__check_output_name(output_name, first_order_indices)
        cls.__check_output_component(
            output_name, output_component, len(first_order_indices[output_name])
        )
        return cls(
            cls.__preprocess(analysis.indices.first[output_name][output_component]),
            cls.__preprocess(analysis.indices.total[output_name][output_component]),
            cls.__preprocess_second_order(
                analysis.indices.second[output_name][output_component]
            ),
        )

    @classmethod
    def from_pce(
        cls, pce_regressor: PCERegressor, output_name: str, output_component: int = 0
    ) -> SobolGraph:
        """Create the Sobol' graph from a PCE (polynomial chaos expansion) regressor.

        Args:
            pce_regressor: A PCE regressor.
            output_name: The name of the output.
            output_component: The component of the output.

        Returns:
            The Sobol' graph associated with this PCE regressor.

        Raises:
            ValueError: When the name is not an output name
                or when the component is out of the range of the output.
        """
        cls.__check_output_name(output_name, pce_regressor.output_names)
        sizes = pce_regressor.sizes
        cls.__check_output_component(output_name, output_component, sizes[output_name])
        index = 0
        for name in pce_regressor.output_names:  # pragma: no branch
            if name == output_name:
                index += output_component
                break

            index += sizes[name]

        return cls(
            cls.__preprocess({
                k: atleast_1d(v).round(3)
                for k, v in pce_regressor.first_sobol_indices[index].items()
            }),
            cls.__preprocess({
                k: atleast_1d(v).round(3)
                for k, v in pce_regressor.total_sobol_indices[index].items()
            }),
            cls.__preprocess_second_order({
                k1: {k2: atleast_2d(v2).round(3) for k2, v2 in v1.items()}
                for k1, v1 in pce_regressor.second_sobol_indices[index].items()
            }),
        )
