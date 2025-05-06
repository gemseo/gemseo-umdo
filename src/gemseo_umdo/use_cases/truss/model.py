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
"""Defining the truss model."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING
from typing import Final

import matplotlib.pyplot as plt
from gemseo.utils.matplotlib_figure import save_show_figure
from numpy import arange
from numpy import array
from numpy import block
from numpy import full
from numpy import ndarray
from numpy import newaxis
from numpy import setdiff1d
from numpy import zeros
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from gemseo_umdo.use_cases.truss.bars import Bars
from gemseo_umdo.use_cases.truss.forces import Forces

if TYPE_CHECKING:
    from pathlib import Path


class TrussModel:
    """The truss model."""

    __ELEMENTS: Final[tuple[tuple[int, int], ...]] = (
        # Oblique bars
        (0, 7),
        (7, 1),
        (1, 8),
        (8, 2),
        (2, 9),
        (9, 3),
        (3, 10),
        (10, 4),
        (4, 11),
        (11, 5),
        (5, 12),
        (12, 6),
        # Bottom horizontal bars
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        # Top horizontal bars
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
    )
    """The elements defined by the indices of the nodes defining their extremities."""

    __NODES: Final[ndarray] = array([
        # Bottom nodes
        [0.0, 0.0],
        [4.0, 0.0],
        [8.0, 0.0],
        [12.0, 0.0],
        [16.0, 0.0],
        [20.0, 0.0],
        [24.0, 0.0],
        # Top nodes
        [2.0, 2.0],
        [6.0, 2.0],
        [10.0, 2.0],
        [14.0, 2.0],
        [18.0, 2.0],
        [22.0, 2.0],
    ])
    """The position of the nodes in x and y directions (in m)."""

    __N_DOF: Final[int] = len(__NODES) * 2
    """The number of degrees of freedom."""

    __FREE_DOF: Final[ndarray] = setdiff1d(arange(__N_DOF), [0, 1, 13])
    """The free degrees of freedom."""

    def compute(
        self,
        bars: Bars | None = None,
        forces: Forces | None = None,
    ) -> tuple[float, ndarray]:
        """Compute the vertical displacements.

        Args:
            bars: The settings of the bars.
                If `None`, the model uses default values.
            forces: The settings of the forces.
                If `None`, the model uses default values.

        Returns:
            The vertical displacement at the bottom central node,
            and the nodal displacements, shaped as `(n_nodes, 2)`
            (the first and second columns represent the horizontal
            and vertical displacements respectively).
        """
        if bars is None:
            bars = Bars()

        if forces is None:
            forces = Forces()

        forces = block([
            [zeros((7, 2))],
            [zeros((6, 1)), full((6, 1), -array(forces)[:, newaxis])],
        ])
        free_forces = forces.flatten()[self.__FREE_DOF]
        stiffness_matrix = lil_matrix((self.__N_DOF, self.__N_DOF))
        for bar, element in zip(bars, self.__ELEMENTS):
            node_1, node_2 = element
            element_stiffness_matrix = self.__compute_element_stiffness_matrix(
                bar.young_modulus, bar.area, node_1, node_2
            )
            dof_map = [node_1 * 2, node_1 * 2 + 1, node_2 * 2, node_2 * 2 + 1]
            for i in range(4):
                for j in range(4):
                    stiffness_matrix[dof_map[i], dof_map[j]] += (
                        element_stiffness_matrix[i, j]
                    )

        free_stiffness_matrix = stiffness_matrix[self.__FREE_DOF, :][:, self.__FREE_DOF]
        free_stiffness_matrix = csr_matrix(free_stiffness_matrix)

        displacements = zeros(self.__N_DOF)
        displacements[self.__FREE_DOF] = spsolve(free_stiffness_matrix, free_forces)

        return displacements[7], displacements.reshape((-1, 2))

    def __compute_element_stiffness_matrix(
        self,
        young_modulus: float,
        area: float,
        first_node: int,
        second_node: int,
    ) -> ndarray:
        """Compute the stiffness matrix of an element.

        Args:
            young_modulus: The Young's modulus of the element.
            area: The cross-sectional area of the element.
            first_node: The index of the first node of the element.
            second_node: The index of the second node of the element.

        Returns:
            The stiffness matrix of the element.
        """
        x1, y1 = self.__NODES[first_node]
        x2, y2 = self.__NODES[second_node]
        length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        cos_theta = (x2 - x1) / length
        sin_theta = (y2 - y1) / length
        cos_sin_theta = cos_theta * sin_theta
        cos_theta_2 = cos_theta**2
        sin_theta_2 = sin_theta**2
        return (young_modulus * area / length) * array([
            [cos_theta_2, cos_sin_theta, -cos_theta_2, -cos_sin_theta],
            [cos_sin_theta, sin_theta_2, -cos_sin_theta, -sin_theta_2],
            [-cos_theta_2, -cos_sin_theta, cos_theta_2, cos_sin_theta],
            [-cos_sin_theta, -sin_theta_2, cos_sin_theta, sin_theta_2],
        ])

    def plot(
        self,
        displacements: ndarray | None = None,
        scale: float = 1.0,
        show: bool = True,
        file_path: str | Path = "",
    ) -> None:
        """Plot the truss structure.

        Args:
            displacements: The nodal displacements if any,
                shaped as `(n_nodes, 2)`
                (the first and second columns represent the horizontal
                and vertical displacements respectively).
            scale: A scaling factor to better see the displacements.
            show: Whether to display the visualization.
            file_path: The path of the file to save the visualization.
        """
        fig = plt.figure(figsize=(10, 6))

        elements = self.__ELEMENTS
        nodes = self.__NODES

        for element in elements:
            first_node, second_node = element
            x = [nodes[first_node, 0], nodes[second_node, 0]]
            y = [nodes[first_node, 1], nodes[second_node, 1]]
            plt.plot(
                x,
                y,
                "b-",
                label="Original elements" if element == elements[0] else None,
            )

        if displacements is not None:
            displaced_nodes = nodes + scale * displacements
            for element in elements:
                first_node, second_node = element
                x = [displaced_nodes[first_node, 0], displaced_nodes[second_node, 0]]
                y = [displaced_nodes[first_node, 1], displaced_nodes[second_node, 1]]
                plt.plot(
                    x,
                    y,
                    "r--",
                    label="Displaced elements" if element == elements[0] else None,
                )

        plt.scatter(nodes[:, 0], nodes[:, 1], color="blue", label="Nodes")
        title = "The truss"
        if displacements is not None:
            plt.scatter(
                displaced_nodes[:, 0],
                displaced_nodes[:, 1],
                color="red",
                marker="o",
                label="Displaced nodes",
            )
            title += " and its displacement"

        plt.title(title)
        plt.xlabel("X-coordinate (m)")
        plt.ylabel("Y-coordinate (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        save_show_figure(fig, show, file_path)
