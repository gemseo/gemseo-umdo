# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""# Take advantage of vectorized disciplines

GEMSEO v6.2 opened the door to batch sampling
by making the MDO formulations support vectorized disciplines
and adding the ``vectorize`` option to the DOE algorithms.
In other words,
GEMSEO can evaluate a multidisciplinary system at several points at the same time,
without the need for multiple processes (see the DOE option ``n_processes`` for more information).
This can be particularly useful when evaluating such a system
in parallel is more expensive than evaluating it serially because
the disciplines are so inexpensive.
In this case, the batch sampling can be sequential.

To illustrate this new feature,
GEMSEO v6.2 vectorizes the disciplines of the Sellar problem
([Sellar1][gemseo.problems.mdo.sellar.sellar_1.Sellar1],
[Sellar2][gemseo.problems.mdo.sellar.sellar_2.Sellar2] and
[SellarSystem][gemseo.problems.mdo.sellar.sellar_system.SellarSystem]).
This example uses them to demonstrate the interest of vectorization
when solving an MDO problem under uncertainty
whose statistics are estimated by Monte Carlo sampling.
"""

from gemseo.algos.doe.scipy.settings.mc import MC_Settings
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.datasets.optimization_dataset import OptimizationDataset
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.uncertainty.distributions.openturns.triangular_settings import (
    OTTriangularDistribution_Settings,
)
from gemseo.utils.timer import Timer

from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

# %%
# First,
# we create a function to solve the MDO problem under uncertainty
# depending on a boolean argument `vectorize`.
# The Monte Carlo sampling to estimate the statistics can be performed
# either sequentially when `vectorize` is `True`
# or all at once when `vectorize` is `False`.


def solve_problem(vectorize: bool) -> tuple[float, OptimizationDataset]:
    """Solve the MDO problem under uncertainty.

    Args:
        vectorize: Whether to enable vectorization.

    Returns:
        The elapsed time and the dataset
        including the value of the design variables, the objective and the constraints
        at each iteration of the algorithm in charge to solve the problem.
    """

    disciplines = [Sellar1(), Sellar2(), SellarSystem()]

    design_space = SellarDesignSpace(dtype="float")

    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable(
        "gamma", OTTriangularDistribution_Settings(minimum=0.1, mode=0.2, maximum=0.3)
    )

    scenario = UMDOScenario(
        disciplines,
        "obj",
        design_space,
        uncertain_space,
        "Mean",
        Sampling_Settings(
            # Note: The default value of vectorize is False, whatever the DOE algorithm.
            doe_algo_settings=MC_Settings(n_samples=100, vectorize=vectorize)
        ),
        formulation_settings_model=MDF_Settings(
            main_mda_settings=MDAGaussSeidel_Settings()
        ),
    )
    scenario.add_constraint("c_1", "Margin")
    scenario.add_constraint("c_2", "Margin")
    with Timer() as timer:
        scenario.execute(MC_Settings(n_samples=20))

    return timer.elapsed_time, scenario.to_dataset()


# %%
# We are ready to solve the MDO problem under uncertainty sequentially,
# in order to get a reference:
time_no_vect, dataset_no_vect = solve_problem(False)

# %%
# Now,
# we can solve this problem using batch sampling:
time_vect, dataset_vect = solve_problem(True)

# %%
# We see that both executions produce similar logs.
# In particular,
# the objective value is the same at each iteration... which is reassuring!
# This is confirmed by checking that all results are equal:
dataset_vect.equals(dataset_vect)

# %%
# That's reassuring, isn't it?
#
# Last but not least,
# we can look at the execution time
# when sampling the process in batch mode (`vectorize=True`):
round(time_vect, 2)

# %%
# and see that it is much lower than in sequential mode (`vectorize=False`):
round(time_no_vect, 2)

# %%
# More precisely,
# the time execution is reduced by
print(round((time_no_vect - time_vect) / time_no_vect * 100), "%")
