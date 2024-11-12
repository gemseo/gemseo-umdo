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
from typing import Any
from typing import Callable
from typing import ClassVar

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.doe.scipy.scipy_doe import SciPyDOE
from gemseo.mlearning.regression.quality.factory import RegressorQualityFactory
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.logging_tools import LoggingContext
from gemseo.utils.seeder import SEED
from numpy import full

from gemseo_umdo.formulations._functions.statistic_function_for_surrogate import (
    StatisticFunctionForSurrogate,
)
from gemseo_umdo.formulations._statistics.sampling.factory import (
    SamplingEstimatorFactory,
)
from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation
from gemseo_umdo.formulations.surrogate_settings import SurrogateSettings

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

    Settings: ClassVar[type[SurrogateSettings]] = SurrogateSettings

    __doe_algo: BaseDOELibrary
    """The DOE library to execute the DOE algorithm."""

    __doe_algo_options: dict[str, Any]
    """The options of the DOE algorithm."""

    __n_samples: int | None
    """The number of samples, if defined."""

    cv_threshold: dict[str, RealArray]
    """The cross-validation threshold component-wise."""

    input_samples: dict[str, RealArray]
    """The input_samples."""

    is_surrogate_quality_bad: Callable[[float, float], bool]
    """A function to indicate if the regressor is good for an output component."""

    regressor_name: str
    """The regressor class name."""

    regressor_options: StrKeyMapping
    """The
    [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor]."""

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

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: BaseMDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        objective_statistic_parameters: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        doe_algo: str = "OT_OPT_LHS",
        doe_algo_options: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        doe_n_samples: int | None = None,
        regressor_name: str = "RBFRegressor",
        regressor_options: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        regressor_n_samples: int = 10000,
        regressor_sampling_seed: int = SEED,
        quality_name: str = "R2Measure",
        quality_threshold: float | Mapping[str, float | Iterable[float]] = 0.9,
        quality_cv_compute: bool = True,
        quality_cv_n_folds: int = 5,
        quality_cv_randomize: bool = True,
        quality_cv_seed: int | None = None,
        quality_cv_threshold: float | Mapping[str, float | Iterable[float]] = 0.8,
        mdo_formulation_settings: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        **settings: Any,
    ) -> None:
        """
        Args:
            doe_n_samples: The number of samples to be generated by the DOE algorithm.
                If `None`,
                the DOE algorithm does not use `doe_n_samples` argument
                but potentially a mandatory argument to be defined in `doe_algo_options`
                (e.g. `samples` for the `CustomDOE` algorithm).
            doe_algo: The name of the DOE algorithm.
            doe_algo_options: The options of the DOE algorithm.
            regressor_name: The regressor class name.
            regressor_options: The options of the
                [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
            regressor_n_samples: The number of Monte Carlo samples
                to estimate the statistics from the regressor.
            regressor_sampling_seed: the seed of the Monte Carlo sampler.
            quality_threshold: The learning quality threshold
                below which a warning is logged.
            quality_name: The name of the measure
                to assess the quality of the regressor.
            quality_cv_compute: Whether to estimate the quality
                by cross-validation (CV).
            quality_n_folds: The number of folds in the case of the CV technique.
            quality_cv_randomize: Whether to shuffle the samples
                before dividing them in folds in the case of the CV technique.
            quality_cv_seed: The seed of the pseudo-random number generator.
                If ``None``,
                an unpredictable generator is used.
            quality_cv_threshold: The CV quality threshold
                below which a warning is logged.

        Raises:
            ValueError: When `n_samples` is `None`,
                whereas it is required by the DOE algorithm.
        """  # noqa: D205 D212 D415
        self.input_data_to_output_samples = {}
        self.__doe_algo = DOELibraryFactory().create(doe_algo)
        self.__doe_algo_options = dict(doe_algo_options)
        model_fields = self.__doe_algo.ALGORITHM_INFOS[doe_algo].Settings.model_fields
        if "n_samples" in model_fields:
            if doe_n_samples is None:
                msg = (
                    "The doe_n_samples argument of "
                    f"the U-MDO formulation '{self.__class__.__name__}' "
                    "is required."
                )
                raise ValueError(msg)
            self.__doe_algo_options["n_samples"] = doe_n_samples

        if "seed" in model_fields and self.__doe_algo_options.get("seed") is None:
            # Use a constant seed to use the same input samples at each design value.
            self.__doe_algo_options["seed"] = SEED

        self.__n_samples = doe_n_samples
        self._estimators = []
        self.regressor_name = regressor_name
        self.regressor_options = regressor_options
        self.input_samples = uncertain_space.convert_array_to_dict(
            SciPyDOE("MC").compute_doe(
                uncertain_space,
                n_samples=regressor_n_samples,
                seed=regressor_sampling_seed,
            )
        )
        self.quality = RegressorQualityFactory().get_class(quality_name)
        if quality_cv_compute:
            self.quality_cv_options = {
                "n_folds": quality_cv_n_folds,
                "seed": quality_cv_seed,
                "randomize": quality_cv_randomize,
            }
        else:
            self.quality_cv_options = {}

        self.quality_threshold = quality_threshold
        self.quality_cv_threshold = quality_cv_threshold
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            mdo_formulation,
            uncertain_space,
            objective_statistic_name,
            objective_statistic_parameters=objective_statistic_parameters,
            mdo_formulation_settings=mdo_formulation_settings,
            **settings,
        )
        mdo_formulation = self._mdo_formulation.__class__.__name__
        formulation = self.__class__.__name__
        self.name = f"{formulation}[{mdo_formulation}; {doe_algo}({doe_n_samples})]"
        smaller_is_better = self.quality.SMALLER_IS_BETTER
        self.is_surrogate_quality_bad = gt if smaller_is_better else lt
        self.quality_operators = ("<=", ">") if smaller_is_better else (">=", "<")
        self.threshold = {}
        self.cv_threshold = {}

    def compute_samples(self, problem: OptimizationProblem) -> IODataset:
        """Evaluate the functions of a problem with a DOE algorithm.

        Args:
            problem: The problem.
        """
        with LoggingContext(logging.getLogger("gemseo")):
            self.__doe_algo.execute(problem, **self.__doe_algo_options)

        io_dataset = problem.to_dataset(opt_naming=False)
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
