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
from typing import ClassVar

from gemseo.post._graph_view import GraphView
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from gemseo.uncertainty.sensitivity.sobol.analysis import SobolAnalysis
    from numpy.typing import NDArray


class SobolGraph(GraphView):
    """A network of uncertain variables representing their Sobol' indices.

    A node represents an uncertain variable whose name is written inside, followed by
    its first-order and total-order Sobol' indices.

    The thickness of a node is proportional to the total-order Sobol' index of the
    variable while the thickness of an edge is proportional to the second-order Sobol'
    index of the corresponding pair of variables.
    """

    DEFAULT_FILE_PATH: ClassVar[str | Path] = "sobol_graph.png"
    """The default file path to save the graph."""

    threshold: float
    """The threshold above which an edge is significant."""

    def __init__(
        self,
        first_order_indices: Mapping[str, float],
        total_order_indices: Mapping[str, float],
        second_order_indices: Mapping[tuple[str, str], float],
        threshold: float = 0.1,
        maximum_thickness: float = 10.0,
    ) -> None:
        """
        Args:
            first_order_indices: The first-order Sobol' indices of the scalar inputs,
                shaped as `{name: index}`.
            second_order_indices:  The second-order Sobol' indices of the scalar inputs,
                shaped as `{(name, other_name): index}`.
            total_order_indices: The total-order Sobol' indices of the scalar inputs,
                shaped as `{name: index}`.
            threshold: The sensitivity threshold
                above which a second-order index is significant
                and the corresponding edge plotted.
            maximum_thickness: The maximum thickness of a line.
        """  # noqa: D205 D212 D415
        super().__init__(False)
        variables_to_nodes = {}

        # Add the nodes representing both first- and total-order indices.
        for name, total_order_index in total_order_indices.items():
            first_order_index = first_order_indices[name]
            node_name = (
                f"{name}\n"
                f"({round(total_order_index * 100)}, {round(first_order_index * 100)})"
            )
            variables_to_nodes[name] = node_name
            self.node(
                node_name,
                penwidth=str(total_order_index * maximum_thickness),
            )

        # Add the edges representing the second-order indices.
        for (name, other_name), index in second_order_indices.items():
            if index >= threshold:
                self.edge(
                    variables_to_nodes[name],
                    variables_to_nodes[other_name],
                    penwidth=str(index * maximum_thickness),
                )

    @staticmethod
    def __preprocess(indices: dict[str, NDArray[float]]) -> dict[str, float]:
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
        cls, indices: dict[str, dict[str, NDArray[float]]]
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
        """
        return cls(
            cls.__preprocess(analysis.indices.first[output_name][output_component]),
            second_order_indices=cls.__preprocess_second_order(
                analysis.indices.second[output_name][output_component]
            ),
            total_order_indices=cls.__preprocess(
                analysis.indices.total[output_name][output_component]
            ),
        )
