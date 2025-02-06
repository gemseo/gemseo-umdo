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
r"""Surrogate-based U-MDO formulation.

[Surrogate][gemseo_umdo.formulations.surrogate.Surrogate] is an
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
estimating the statistics with (quasi) Monte Carlo techniques
applied to a surrogate model.

E.g.

$$\mathbb{E}[f(x,U)] \approx \frac{1}{N}\sum_{i=1}^N \hat{f}_x\left(U^{(i)}\right)$$

or

$$\mathbb{V}[f(x,U)] \approx
\frac{1}{N}\sum_{i=1}^N \left(\hat{f}_x\left(U^{(i)}\right)-
\frac{1}{N}\sum_{j=1}^N \hat{f}_x\left(U^{(j)}\right)\right)^2$$

where $\hat{f}_x$ is a surrogate model of $f$ at $x$,
$U$ is normally distributed
with mean $\mu$ and variance $\sigma^2$
and $U^{(1)},\ldots,U^{(N)}$ are $N$ realizations of $U$
obtained with an optimized Latin hypercube sampling technique.
"""

from __future__ import annotations

import logging
from numbers import Number
from operator import gt
from operator import lt
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.doe.scipy.scipy_doe import SciPyDOE
from gemseo.mlearning.regression.quality.factory import RegressorQualityFactory
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.logging_tools import LoggingContext
from numpy import full

from gemseo_umdo.formulations._functions.statistic_function_for_surrogate import (
    StatisticFunctionForSurrogate,
)
from gemseo_umdo.formulations._statistics.sampling.factory import (
    SamplingEstimatorFactory,
)
from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation
from gemseo_umdo.formulations.surrogate_settings import Surrogate_Settings

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe.base_doe_library import BaseDOELibrary
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
    from gemseo.mlearning.regression.quality.base_regressor_quality import (
        BaseRegressorQuality,
    )
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping


class Surrogate(BaseUMDOFormulation):
    """Surrogate-based U-MDO formulation.

    !!! note "DOE algorithms"
        This formulation uses a DOE algorithm;
        read the
        [GEMSEO documentation](https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html).
        for more information about the available DOE algorithm names and options.

    !!! note "Regression algorithms"
        This formulation uses a regression algorithm;
        read the
        [GEMSEO documentation](https://gemseo.readthedocs.io/en/stable/algorithms/surrogate_algos.html).
        for more information about the available regression algorithm names and options.
    """

    Settings: ClassVar[type[Surrogate_Settings]] = Surrogate_Settings

    __doe_algo: BaseDOELibrary
    """The DOE library to execute the DOE algorithm."""

    cv_threshold: dict[str, RealArray]
    """The cross-validation threshold component-wise."""

    input_samples: dict[str, RealArray]
    """The input_samples."""

    is_surrogate_quality_bad: Callable[[float, float], bool]
    """A function to indicate if the regressor is good for an output component."""

    quality: type[BaseRegressorQuality]
    """The class to assess the quality of the regressor."""

    quality_threshold: float | Mapping[str, float | Iterable[float]]
    """The learning quality threshold below which a warning is logged."""

    quality_cv_options: dict[str, bool | int | None]
    """The options of the CV technique; if empty, do not use it."""

    quality_cv_threshold: float | Mapping[str, float | Iterable[float]]
    """The CV quality threshold below which a warning is logged."""

    quality_operators: tuple[str, str]
    """The operators ``(o1, o2)`` to compare the quality of the regressor and a
    threshold.

    "A o1 B" means that A is better or equal to B. "A o2 B" means that A is less good
    than B.
    """

    threshold: dict[str, RealArray]
    """The learning quality threshold component-wise."""

    _STATISTIC_FACTORY: ClassVar[SamplingEstimatorFactory] = SamplingEstimatorFactory()

    _STATISTIC_FUNCTION_CLASS: ClassVar[type[StatisticFunctionForSurrogate] | None] = (
        StatisticFunctionForSurrogate
    )

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: BaseMDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        settings_model: Surrogate_Settings,
        objective_statistic_parameters: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        mdo_formulation_settings: StrKeyMapping = READ_ONLY_EMPTY_DICT,
    ) -> None:
        algo_name = settings_model.doe_algo_settings._TARGET_CLASS_NAME
        self.input_data_to_output_samples = {}
        self.__doe_algo = DOELibraryFactory().create(algo_name)
        self._estimators = []
        self.input_samples = uncertain_space.convert_array_to_dict(
            SciPyDOE("MC").compute_doe(
                uncertain_space,
                n_samples=settings_model.regressor_n_samples,
                seed=settings_model.regressor_sampling_seed,
            )
        )
        self.quality = RegressorQualityFactory().get_class(settings_model.quality_name)
        if settings_model.quality_cv_compute:
            self.quality_cv_options = {
                "n_folds": settings_model.quality_cv_n_folds,
                "seed": settings_model.quality_cv_seed,
                "randomize": settings_model.quality_cv_randomize,
            }
        else:
            self.quality_cv_options = {}

        self.quality_threshold = settings_model.quality_threshold
        self.quality_cv_threshold = settings_model.quality_cv_threshold
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            mdo_formulation,
            uncertain_space,
            objective_statistic_name,
            settings_model,
            objective_statistic_parameters=objective_statistic_parameters,
            mdo_formulation_settings=mdo_formulation_settings,
        )
        mdo_formulation = self._mdo_formulation.__class__.__name__
        formulation = self.__class__.__name__
        smaller_is_better = self.quality.SMALLER_IS_BETTER
        doe_n_samples = settings_model.n_samples
        self.name = f"{formulation}[{mdo_formulation}; {algo_name}({doe_n_samples})]"
        self.is_surrogate_quality_bad = gt if smaller_is_better else lt
        self.quality_operators = ("<=", ">") if smaller_is_better else (">=", "<")
        self.threshold = {}
        self.cv_threshold = {}

    def compute_samples(
        self, problem: OptimizationProblem, compute_jacobian: bool = False
    ) -> IODataset:
        """Evaluate the functions of a problem with a DOE algorithm.

        Args:
            problem: The problem.
            compute_jacobian: Whether to compute the Jacobian of the objective.
        """
        doe_algo_settings = self._settings.doe_algo_settings
        doe_algo_settings.eval_jac = compute_jacobian
        with LoggingContext(logging.getLogger("gemseo")):
            self.__doe_algo.execute(
                problem, eval_obs_jac=compute_jacobian, settings_model=doe_algo_settings
            )

        io_dataset = problem.to_dataset(opt_naming=False, export_gradients=True)
        if not self.threshold:
            names_to_sizes = {
                name: len(
                    io_dataset.get_variable_components(io_dataset.OUTPUT_GROUP, name)
                )
                for name in io_dataset.output_names
            }
            self.threshold = self.__compute_threshold(
                names_to_sizes, self.quality_threshold
            )
            self.cv_threshold = self.__compute_threshold(
                names_to_sizes, self.quality_cv_threshold
            )

        return io_dataset

    @staticmethod
    def __compute_threshold(
        names_to_sizes: Mapping[str, int],
        user_threshold: float | Mapping[str, float | RealArray],
    ) -> dict[str, RealArray]:
        """Compute the quality threshold component-wise.

        Args:
            names_to_sizes: The names and sizes of the output variables.
            user_threshold: The quality threshold given by the user.

        Returns:
            The quality threshold component-wise.
        """
        if isinstance(user_threshold, Number):
            threshold = {
                name: full(size, user_threshold)
                for name, size in names_to_sizes.items()
            }
        else:
            threshold = {name: full(size, 0.8) for name, size in names_to_sizes.items()}
            for name, value in user_threshold.items():
                threshold[name][:] = value

        return threshold
