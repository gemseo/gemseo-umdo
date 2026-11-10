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
"""A function to compute a statistic from `Sampling`.

See also [Sampling][gemseo_umdo.formulation.sampling.Sampling].
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import TypeAlias
from typing import TypeVar

from numpy import array
from numpy import array_equal
from numpy import flatnonzero

from gemseo_umdo.formulation._function.base import BaseStatisticFunction
from gemseo_umdo.formulation._function.base import _with_jacobian_name
from gemseo_umdo.formulation._function.iterative_estimation import IterativeEstimation

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.core.problem.evaluation import EvaluationProblem
    from gemseo.util.typing import RealArray

    from gemseo_umdo.formulation.sampling import Sampling
    from gemseo_umdo.statistic.settings import BaseStatisticSettings

SamplingT = TypeVar("SamplingT", bound="Sampling")

_EstimationData: TypeAlias = tuple["RealArray | None", int]
"""The iterative estimation of a statistic or of its Jacobian, if any,
and the number of samples used to compute it."""

logger = logging.getLogger(__name__)


class StatisticFunctionForSampling(BaseStatisticFunction[SamplingT]):
    """A function to compute a statistic from `Sampling`.

    The statistic is estimated either iteratively,
    by updating its estimation after each evaluation of the output,
    or in one go from the samples of the output,
    according to the estimation mode of the U-MDO formulation.

    The function takes part in the sampling of the U-MDO formulation
    once created or added by this formulation.
    """

    _can_estimate_jacobian: ClassVar[bool] = True

    iterative_estimation: IterativeEstimation | None
    """The iterative estimation of the statistic and of its Jacobian, if any.

    `None` when the U-MDO formulation does not estimate the statistics iteratively.
    """

    def __init__(  # noqa: D107
        self,
        umdo_formulation: SamplingT,
        output_name: str,
        function_type: ArrayFunction.FunctionType,
        statistic_settings: BaseStatisticSettings,
    ) -> None:
        super().__init__(
            umdo_formulation,
            output_name,
            function_type,
            statistic_settings,
        )
        self.iterative_estimation = None
        if umdo_formulation._estimate_statistics_iteratively:
            self.iterative_estimation = IterativeEstimation(
                output_name,
                self._statistic_name,
                self._statistic_estimator,
                self._statistic_jac_name,
            )

    @property
    def __estimate_iteratively(self) -> bool:
        """Whether to estimate the statistic iteratively."""
        return self.iterative_estimation is not None

    def _compute_statistic_estimation(self, data: Mapping[str, Any]) -> RealArray:
        with_gradient = data[_with_jacobian_name]
        if self.__estimate_iteratively:
            estimation, n_samples = data[self._statistic_name]
            self._check_n_samples(n_samples, with_gradient)
            return estimation

        samples = data.get(self._output_name)
        self._check_n_samples(self.__get_n_samples(samples), with_gradient)
        return self._statistic_estimator.estimate_statistic(samples)

    def _compute_statistic_jacobian_estimation(
        self, data: Mapping[str, Any]
    ) -> RealArray:
        if self.__estimate_iteratively:
            estimation, n_samples = data[self._statistic_jac_name]
            self._check_n_samples(n_samples, True, jacobian=True)
            return estimation

        samples = data.get(self._output_name)
        self._check_n_samples(self.__get_n_samples(samples), True, jacobian=True)
        return self._statistic_estimator.compute_jacobian(
            samples, data[self._output_jac_name]
        )

    @staticmethod
    def __get_n_samples(samples: RealArray | None) -> int:
        """Return the number of samples.

        Args:
            samples: The samples, if any;
                an output missing at all the samples has no entry in the data.

        Returns:
            The number of samples.
        """
        return 0 if samples is None else len(samples)

    def _check_n_samples(
        self, n_samples: int, with_gradient: bool, jacobian: bool = False
    ) -> None:
        """Check that the number of samples suffices to estimate the statistic.

        Args:
            n_samples: The number of samples used to estimate the statistic
                or its Jacobian.
            with_gradient: Whether these samples are those
                where both the output of interest and its gradient are available.
            jacobian: Whether to check the samples
                to estimate the Jacobian of the statistic.

        Raises:
            ValueError: When the output of interest
                or, with `with_gradient`, its gradient
                is not available at any sample of the uncertain variables,
                or when they are available at fewer samples
                than the statistic requires.
        """
        output_name = self._output_name
        if not n_samples:
            statistics = "Jacobian of the statistics" if jacobian else "statistics"
            if with_gradient:
                missing = f"{output_name} and its gradient are not available together"
                causes = (
                    "their evaluation failed "
                    "or the DOE algorithm does not store the Jacobian"
                )
            else:
                missing = f"{output_name} is not available"
                causes = "its evaluation failed"

            msg = (
                f"The {statistics} of {output_name} cannot be estimated, "
                f"as {missing} at any sample of the uncertain variables, "
                f"e.g. because {causes}."
            )
            raise ValueError(msg)

        min_n_samples = self._statistic_estimator.min_n_samples
        if n_samples >= min_n_samples:
            return

        statistic = f"statistic {self._statistic_name}"
        if jacobian:
            statistic = f"Jacobian of the {statistic}"

        names = output_name
        where = f"of {output_name}"
        pronoun = "its"
        if with_gradient:
            names = f"{output_name} and its gradient"
            where = "where both are available"
            pronoun = "their"

        msg = (
            f"The {statistic} cannot be estimated, "
            f"as it requires at least {min_n_samples} samples of {names}, "
            f"while the number of samples {where} is {n_samples}, "
            "e.g. because the DOE algorithm generated too few samples "
            "of the uncertain variables "
            f"or because {pronoun} evaluation failed at some of these samples."
        )
        raise ValueError(msg)

    @staticmethod
    def __log_missing_samples(
        n_doe_samples: int,
        output_names_to_n_samples: Mapping[str, int],
        with_gradient: bool,
    ) -> None:
        """Log the outputs missing at some samples of the uncertain variables.

        An output is reported as `"1 for f"`
        when its statistics are estimated from fewer samples
        than generated by the DOE algorithm,
        or as `"1 for f and its gradient"`
        when its statistics and their Jacobians are estimated
        from the samples where both are available.

        Args:
            n_doe_samples: The number of samples generated by the DOE algorithm.
            output_names_to_n_samples: The names of the outputs
                bound to the number of samples used to estimate their statistics.
                If no output is missing at any sample, do not log anything.
            with_gradient: Whether these samples are those
                where both the outputs and their gradients are available.
        """
        suffix = " and its gradient" if with_gradient else ""
        missing_samples = [
            f"{n_samples} for {output_name}{suffix}"
            for output_name, n_samples in output_names_to_n_samples.items()
            if n_samples < n_doe_samples
        ]
        if missing_samples:
            # A single message for all the outputs avoids flooding the log.
            logger.warning(
                "Some outputs or gradients are missing at some of the %s samples "
                "of the uncertain variables, e.g. because their evaluation failed; "
                "the numbers of samples used to estimate the statistics are: %s.",
                n_doe_samples,
                ", ".join(missing_samples),
            )

    def _compute_data_for_statistic_estimation(
        self, input_data: RealArray, estimate_jacobian: bool
    ) -> dict[str, Any]:
        formulation = self._umdo_formulation
        problem = formulation.mdo_formulation.problem
        formulation.compute_samples(
            problem, input_data, compute_jacobian=estimate_jacobian
        )
        doe_samples = formulation.doe_algo.samples
        if self.__estimate_iteratively:
            return self.__get_iterative_estimations(
                formulation._iterative_estimations,
                len(doe_samples),
                problem.function_names,
                estimate_jacobian,
            )

        return self.__get_samples(problem, doe_samples, estimate_jacobian)

    @classmethod
    def __get_iterative_estimations(
        cls,
        estimations: Iterable[IterativeEstimation],
        n_doe_samples: int,
        output_names: Iterable[str],
        estimate_jacobian: bool,
    ) -> dict[str, _EstimationData]:
        """Return the iterative estimations with their numbers of samples.

        Args:
            estimations: The iterative estimations updated by the last sampling.
            n_doe_samples: The number of samples generated by the DOE algorithm.
            output_names: The names of the outputs of the sampled problem,
                in the order of the log.
            estimate_jacobian: Whether the last sampling evaluated the Jacobians.

        Returns:
            The estimations of the statistics and, if any, of their Jacobians,
            bound to the names of these statistics or Jacobians,
            with the numbers of samples used to compute them.
        """
        data = {}
        # The numbers of samples of each output, as passed to __log_missing_samples;
        # the estimations of the statistics of an output use the same samples.
        output_names_to_n_samples = {}
        for estimation in estimations:
            n_samples = estimation.n_samples
            data[estimation.output_statistic_name] = (
                estimation.last_estimation,
                n_samples,
            )
            if (
                estimate_jacobian
                and estimation.statistic_estimator.can_compute_jacobian
            ):
                data[estimation.output_statistic_jac_name] = (
                    estimation.last_jacobian_estimation,
                    n_samples,
                )

            output_names_to_n_samples[estimation.output_name] = n_samples

        # The outputs are logged in the same order as with the one-shot estimation,
        # rather than in the order in which the functions take part in the sampling.
        cls.__log_missing_samples(
            n_doe_samples,
            {
                name: output_names_to_n_samples[name]
                for name in output_names
                if name in output_names_to_n_samples
            },
            estimate_jacobian,
        )
        return data

    @classmethod
    def __get_samples(
        cls,
        problem: EvaluationProblem,
        doe_samples: RealArray,
        estimate_jacobian: bool,
    ) -> dict[str, RealArray]:
        """Return the samples of the outputs and, if required, of their gradients.

        Args:
            problem: The problem sampled by the DOE algorithm.
            doe_samples: The samples generated by the DOE algorithm.
            estimate_jacobian: Whether to return the samples of the gradients.

        Returns:
            The samples of the outputs,
            shaped as `(n_samples, output_size)`,
            and, bound to the names of their gradients if required,
            the samples of these gradients,
            shaped as `(n_samples, output_size, input_size)`.
            When the gradients are required,
            the samples of an output are those where its gradient is available too,
            so that its statistics and their Jacobians use the same samples.
            An output missing at all the samples has no entry.
        """
        # The samples are read from the database in a single pass
        # over those generated by the DOE algorithm,
        # as these data are shared by all the statistic functions.
        # The DOE algorithm works on the uncertain space, which cannot be normalized,
        # so that its samples are the keys of the database,
        # which is reset at each input point.
        database = problem.database
        input_history = database.get_x_vect_history()
        if len(input_history) == len(doe_samples) and array_equal(
            input_history, doe_samples
        ):
            # Each generated sample is an entry of the database, in the same order,
            # so that the entries are read without hashing the samples.
            samples = list(database.values())
        else:
            # The database is keyed by the input point,
            # so that a repeated sample is stored once;
            # looking up each generated sample keeps its weight in the statistics.
            samples = [database.get(input_value) or {} for input_value in doe_samples]

        output_names = problem.function_names
        gradient_names = [database.get_gradient_name(name) for name in output_names]
        names = output_names
        if estimate_jacobian:
            names = [*output_names, *gradient_names]

        # Whether a name has a value at each sample.
        name_to_mask = {
            name: array([name in sample for sample in samples], dtype=bool)
            for name in names
        }
        data = {}
        # The numbers of samples of each output, as passed to __log_missing_samples.
        output_names_to_n_samples = {}
        for output_name, gradient_name in zip(
            output_names, gradient_names, strict=True
        ):
            # The DOE algorithm skips a sample whose evaluation fails,
            # but the database may keep the outputs evaluated before the failure.
            # When the gradients are required,
            # discarding a sample missing the gradient of an output
            # estimates the statistics of this output and their Jacobians
            # from the same samples,
            # as the iterative estimation does,
            # while the statistics of an output do not depend on the failures
            # of the other ones.
            mask = name_to_mask[output_name]
            if estimate_jacobian:
                mask = mask & name_to_mask[gradient_name]

            n_samples = int(mask.sum())
            output_names_to_n_samples[output_name] = n_samples
            if n_samples == 0:
                continue

            indices = flatnonzero(mask)
            # An output has the same size at all the samples,
            # so that its history cannot be ragged.
            output_history = array([
                samples[index][output_name] for index in indices
            ]).reshape(n_samples, -1)
            data[output_name] = output_history
            if not estimate_jacobian:
                continue

            # The gradient of an output of size m
            # is stored as an array shaped as (m, d), or (d,) when m is 1,
            # so that its history is reshaped from the size of the output
            # rather than from the number of dimensions of the stored arrays.
            data[gradient_name] = array([
                samples[index][gradient_name] for index in indices
            ]).reshape(n_samples, output_history.shape[1], -1)

        cls.__log_missing_samples(
            len(doe_samples), output_names_to_n_samples, estimate_jacobian
        )
        return data
