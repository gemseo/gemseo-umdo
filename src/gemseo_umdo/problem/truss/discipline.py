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
"""Defining the truss model as a discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.discipline import Discipline
from numpy import array

from gemseo_umdo.problem.truss.bar import Bar
from gemseo_umdo.problem.truss.bars import Bars
from gemseo_umdo.problem.truss.forces import Forces
from gemseo_umdo.problem.truss.model import TrussModel

if TYPE_CHECKING:
    from gemseo import StrKeyMapping
    from gemseo.core.grammar.factory import GrammarType


class TrussDiscipline(Discipline):
    """The discipline wrapping the truss model."""

    default_grammar_type: ClassVar[GrammarType] = Discipline.GrammarType.SIMPLE

    __use_different_bars: bool
    """Whether the bars are different."""

    __truss_model: TrussModel
    """The truss model."""

    def __init__(self, use_different_bars: bool = False) -> None:
        """
        Args:
            use_different_bars: Whether the bars are different.
                Otherwise, use identical oblical bars and identical horizontal bars.
        """  # noqa: D205 D212
        super().__init__()
        e1_names = (
            tuple(f"E1_{i}" for i in range(1, 12)) if use_different_bars else ("E1",)
        )
        e2_names = (
            tuple(f"E2_{i}" for i in range(1, 13)) if use_different_bars else ("E2",)
        )
        a1_names = (
            tuple(f"A1_{i}" for i in range(1, 12)) if use_different_bars else ("A1",)
        )
        a2_names = (
            tuple(f"A2_{i}" for i in range(1, 13)) if use_different_bars else ("A2",)
        )
        p_names = tuple(f"P{i}" for i in range(1, 7))
        self.io.input_grammar.update_from_names((
            *e1_names,
            *e2_names,
            *a1_names,
            *a2_names,
            *p_names,
        ))
        self.io.input_grammar.descriptions.update({
            name: f"The Young's modulus of the horizontal bar #{i} (in Pa)."
            for i, name in enumerate(e1_names, start=1)
        })
        self.io.input_grammar.descriptions.update({
            name: f"The Young's modulus of the vertical bar #{i} (in Pa)."
            for i, name in enumerate(e2_names, start=1)
        })
        self.io.input_grammar.descriptions.update({
            name: f"The cross-sectional area of the horizontal bar #{i} (in m²)."
            for i, name in enumerate(a1_names, start=1)
        })
        self.io.input_grammar.descriptions.update({
            name: f"The cross-sectional area of the vertical bar #{i} (in m²)."
            for i, name in enumerate(a2_names, start=1)
        })
        self.io.input_grammar.descriptions.update({
            name: f"The vertical force #{i} (in N)."
            for i, name in enumerate(p_names, start=1)
        })
        self.io.output_grammar.update_from_names(("V1", "displacements"))
        self.io.output_grammar.descriptions.update({
            "V1": "The vertical displacement (in m) at the bottom central node.",
            "displacements": (
                "The nodal displacements (in m) shaped as `(n_nodes, 2)` "
                "(the first and second columns "
                "represent the horizontal and vertical displacements respectively)."
            ),
        })
        self.default_input_data = (
            dict.fromkeys(e1_names, array([210000000000.0]))
            | dict.fromkeys(e2_names, array([210000000000.0]))
            | dict.fromkeys(a1_names, array([0.002]))
            | dict.fromkeys(a2_names, array([0.001]))
            | dict.fromkeys(p_names, array([5e4]))
        )
        self.__truss_model = TrussModel()
        self.__use_different_bars = use_different_bars

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        if self.__use_different_bars:
            e1_1 = input_data["E1_1"][0]
            e1_2 = input_data["E1_2"][0]
            e1_3 = input_data["E1_3"][0]
            e1_4 = input_data["E1_4"][0]
            e1_5 = input_data["E1_5"][0]
            e1_6 = input_data["E1_6"][0]
            e1_7 = input_data["E1_7"][0]
            e1_8 = input_data["E1_8"][0]
            e1_9 = input_data["E1_9"][0]
            e1_10 = input_data["E1_10"][0]
            e1_11 = input_data["E1_11"][0]
            e2_1 = input_data["E2_1"][0]
            e2_2 = input_data["E2_2"][0]
            e2_3 = input_data["E2_3"][0]
            e2_4 = input_data["E2_4"][0]
            e2_5 = input_data["E2_5"][0]
            e2_6 = input_data["E2_6"][0]
            e2_7 = input_data["E2_7"][0]
            e2_8 = input_data["E2_8"][0]
            e2_9 = input_data["E2_9"][0]
            e2_10 = input_data["E2_10"][0]
            e2_11 = input_data["E2_11"][0]
            e2_12 = input_data["E2_12"][0]
            a1_1 = input_data["A1_1"][0]
            a1_2 = input_data["A1_2"][0]
            a1_3 = input_data["A1_3"][0]
            a1_4 = input_data["A1_4"][0]
            a1_5 = input_data["A1_5"][0]
            a1_6 = input_data["A1_6"][0]
            a1_7 = input_data["A1_7"][0]
            a1_8 = input_data["A1_8"][0]
            a1_9 = input_data["A1_9"][0]
            a1_10 = input_data["A1_10"][0]
            a1_11 = input_data["A1_11"][0]
            a2_1 = input_data["A2_1"][0]
            a2_2 = input_data["A2_2"][0]
            a2_3 = input_data["A2_3"][0]
            a2_4 = input_data["A2_4"][0]
            a2_5 = input_data["A2_5"][0]
            a2_6 = input_data["A2_6"][0]
            a2_7 = input_data["A2_7"][0]
            a2_8 = input_data["A2_8"][0]
            a2_9 = input_data["A2_9"][0]
            a2_10 = input_data["A2_10"][0]
            a2_11 = input_data["A2_11"][0]
            a2_12 = input_data["A2_12"][0]
        else:
            a1_1 = a1_2 = a1_3 = a1_4 = a1_5 = a1_6 = a1_7 = a1_8 = a1_9 = a1_10 = (
                a1_11
            ) = input_data["A1"][0]
            a2_1 = a2_2 = a2_3 = a2_4 = a2_5 = a2_6 = a2_7 = a2_8 = a2_9 = a2_10 = (
                a2_11
            ) = a2_12 = input_data["A2"][0]
            e1_1 = e1_2 = e1_3 = e1_4 = e1_5 = e1_6 = e1_7 = e1_8 = e1_9 = e1_10 = (
                e1_11
            ) = input_data["E1"][0]
            e2_1 = e2_2 = e2_3 = e2_4 = e2_5 = e2_6 = e2_7 = e2_8 = e2_9 = e2_10 = (
                e2_11
            ) = e2_12 = input_data["E2"][0]

        bars = Bars(
            bar_0=Bar(young_modulus=e2_1, area=a2_1),
            bar_1=Bar(young_modulus=e2_2, area=a2_2),
            bar_2=Bar(young_modulus=e2_3, area=a2_3),
            bar_3=Bar(young_modulus=e2_4, area=a2_4),
            bar_4=Bar(young_modulus=e2_5, area=a2_5),
            bar_5=Bar(young_modulus=e2_6, area=a2_6),
            bar_6=Bar(young_modulus=e2_7, area=a2_7),
            bar_7=Bar(young_modulus=e2_8, area=a2_8),
            bar_8=Bar(young_modulus=e2_9, area=a2_9),
            bar_9=Bar(young_modulus=e2_10, area=a2_10),
            bar_10=Bar(young_modulus=e2_11, area=a2_11),
            bar_11=Bar(young_modulus=e2_12, area=a2_12),
            bar_12=Bar(young_modulus=e1_1, area=a1_1),
            bar_13=Bar(young_modulus=e1_2, area=a1_2),
            bar_14=Bar(young_modulus=e1_3, area=a1_3),
            bar_15=Bar(young_modulus=e1_4, area=a1_4),
            bar_16=Bar(young_modulus=e1_5, area=a1_5),
            bar_17=Bar(young_modulus=e1_6, area=a1_6),
            bar_18=Bar(young_modulus=e1_7, area=a1_7),
            bar_19=Bar(young_modulus=e1_8, area=a1_8),
            bar_20=Bar(young_modulus=e1_9, area=a1_9),
            bar_21=Bar(young_modulus=e1_10, area=a1_10),
            bar_22=Bar(young_modulus=e1_11, area=a1_11),
        )

        forces = Forces(
            force_7=input_data["P1"][0],
            force_8=input_data["P2"][0],
            force_9=input_data["P3"][0],
            force_10=input_data["P4"][0],
            force_11=input_data["P5"][0],
            force_12=input_data["P6"][0],
        )
        v1, displacements = self.__truss_model.compute(bars=bars, forces=forces)
        return {"V1": array([v1]), "displacements": displacements.ravel()}
