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
"""A factory of noising disciplines."""

from __future__ import annotations

from gemseo.disciplines.factory import DisciplineFactory

from gemseo_umdo.disciplines.base_noiser import BaseNoiser


class NoiserFactory(DisciplineFactory):
    """A factory of noising disciplines."""

    _CLASS = BaseNoiser
    _PACKAGE_NAMES = ("gemseo_umdo.disciplines",)

    __short_names_to_class_names: dict[str, str]
    """The {short name: class_name} mapping."""

    def __init__(self):  # noqa: D107
        super().__init__()
        self.__short_names_to_class_names = {
            self.get_class(class_name).SHORT_NAME: class_name
            for class_name in self.class_names
        }

    def create(
        self,
        noiser_name: str,
        variable_name: str,
        noised_variable_name: str,
        uncertain_variable_name: str,
    ) -> BaseNoiser:
        """
        Args:
            noiser_name: Either the class name or the short name
                of the noising discipline.
            variable_name: The name of the variable to be noised.
            noised_variable_name: The name of the variable once noised.
            uncertain_variable_name: The name of the uncertain variable.
        """  # noqa: D205 D212 D415
        class_names = self.class_names
        if noiser_name in class_names:
            class_name = noiser_name
        else:
            class_name = self.__short_names_to_class_names[noiser_name]

        return super().create(
            class_name,
            variable_name=variable_name,
            noised_variable_name=noised_variable_name,
            uncertain_variable_name=uncertain_variable_name,
        )
