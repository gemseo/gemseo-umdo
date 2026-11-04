<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Changelog titles are:
- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.
-->

# Changelog

All notable changes of this project will be documented here.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0)
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Develop

### Added

- [UEvaluationScenario][gemseo_umdo.scenarios.u_evaluation_scenario.UEvaluationScenario]
  evaluates statistics of outputs over the design space with a DOE algorithm;
  it is the counterpart of
  [EvaluationScenario][gemseo.scenario.evaluation.EvaluationScenario]
  for design under uncertainty:
  the statistics are added as observables with
  [add_observable()][gemseo_umdo.scenarios.base_u_scenario.BaseUScenario.add_observable].
- The design strength [Rd][gemseo_umdo.use_cases.beam_model.core.variables.Rd]
  and the yield strength [Ry][gemseo_umdo.use_cases.beam_model.core.variables.Ry]
  of [AdvancedBeamUncertainSpace][gemseo_umdo.use_cases.beam_model.advanced_uncertain_space.AdvancedBeamUncertainSpace]
  are available in
  [gemseo_umdo.use_cases.beam_model.core.variables][gemseo_umdo.use_cases.beam_model.core.variables].
- [Sampling.doe_algo][gemseo_umdo.formulations.sampling.Sampling.doe_algo]
  is the DOE library used to sample the disciplines.
- The U-MDO formulations
  [Sampling][gemseo_umdo.formulations.sampling.Sampling],
  [SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling]
  and [Surrogate][gemseo_umdo.formulations.surrogate.Surrogate]
  support the statistic `"Quantile"`,
  e.g. `scenario.add_constraint("c", "Quantile", level=0.95)`,
  with analytical derivatives for the first two;
  its parameters are the `level` (default: 0.95)
  and the `method` of `numpy.quantile` (default: `"linear"`);
  the `level` is replaced by `max(level, 1 - level)`
  for an objective to minimize or a negativity inequality constraint
  and by `min(level, 1 - level)`
  for an objective to maximize or a positivity inequality constraint,
  so that the quantile is conservative;
  for an observable or an equality constraint, it is used as is;
  the name of the statistic always includes the level,
  e.g. `"Q[y; 0.95]"`.
  For [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  and [SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling],
  this statistic cannot be estimated iteratively:
  using it requires `estimate_statistics_iteratively=False`,
  otherwise a `ValueError` is raised.
- [BaseUMDOFormulation.estimate_statistic_jacobians][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation.estimate_statistic_jacobians]
  is whether to sample a design point once to estimate both the statistics
  and their Jacobians;
  the U-scenarios set it from their driver
  and the differentiation method of their problem,
  and reset it at the end of each execution with
  [BaseUMDOFormulation.reset_execution_state()][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation.reset_execution_state].

### Changed

- [AdvancedBeamUncertainSpace][gemseo_umdo.use_cases.beam_model.advanced_uncertain_space.AdvancedBeamUncertainSpace]
  raises a `ValueError`
  when a name in `nominal_values` or in the dispersions is not an uncertain variable,
  when a dispersion is not positive,
  or when the lower bound $\mu-\delta$ of `b`, `h`, `t`, `L`, `E`, `Rd` or `Ry`
  is not positive.
- GEMSEO-UMDO requires GEMSEO 7.
- BREAKING CHANGE:
  using a statistic that a U-MDO formulation does not support
  raises a `ValueError` listing the available statistics
  instead of an `ImportError`.
- BREAKING CHANGE:
  [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  and [UOptAsUMDOScenario][gemseo_umdo.problems.uopt_as_umdo_scenario.UOptAsUMDOScenario]
  follow the API of the GEMSEO 7 scenarios:
    - they are instantiated from the disciplines (or the discipline),
      the design space, the uncertain space and the statistic estimation settings,
      e.g. `UMDOScenario(disciplines, design_space, uncertain_space, Sampling_Settings())`;
    - the objective is set after instantiation with
      [add_objective()][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario.add_objective],
      e.g. `scenario.add_objective("f", "Mean")`,
      and maximized with `minimize=False`;
      the arguments `objective_name`, `objective_statistic_name`,
      `objective_statistic_parameters` and `maximize_objective` are removed;
    - the MDO formulation is defined by the argument `formulation_settings`,
      a Pydantic model of settings
      (default: [MDF_Settings][gemseo.formulation.mdf_settings.MDF_Settings]);
      the arguments `formulation_name`, `formulation_settings_model`
      and `**formulation_settings` are removed.
- BREAKING CHANGE: the U-MDO formulations follow the API of the GEMSEO 7 formulations:
  they are instantiated from an optimization problem defined over the design space,
  the disciplines, the settings,
  the uncertain space and the settings of the MDO formulation,
  and create the objective with
  [create_objective()][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation.create_objective].
  Their MDO formulations evaluate the functions over the uncertain space
  with an [EvaluationProblem][gemseo.core.problem.evaluation.EvaluationProblem]
  whose observables include the output of the objective.
  The settings of the U-MDO formulations derive from
  [BaseFormulationSettings][gemseo.formulation.core.base_settings.BaseFormulationSettings].
- BREAKING CHANGE:
  [UncertainCouplingGraph.sample()][gemseo_umdo.visualizations.uncertain_coupling_graph.UncertainCouplingGraph.sample]
  takes the settings of the DOE algorithm,
  e.g. `OT_OPT_LHS_Settings(n_samples=100)`,
  instead of `n_samples`, `algo_name` and `**algo_options`.
- [PCE_Settings][gemseo_umdo.formulations.pce_settings.PCE_Settings]
  raises a `ValueError` when `regressor_settings.learn_jacobian_data` is `True`
  and `approximate_statistics_jacobians` is `False`;
  this combination used to estimate the Jacobian of the statistics
  from the derivatives with respect to the uncertain variables
  instead of the design variables.
- BREAKING CHANGE: the `factor` parameter of the statistic `"Margin"` (default: 2)
  is replaced by its absolute value
  for an objective to minimize or a negativity inequality constraint
  and by the opposite of its absolute value
  for an objective to maximize or a positivity inequality constraint,
  so that the margin is conservative;
  for an observable or an equality constraint, it is used as is,
  whereas it used to be replaced by its absolute value for an equality constraint,
  and by the opposite of its absolute value when `positive=True`.
  The name of the statistic always includes the factor,
  e.g. `"Margin[o; 2.0]"` for `add_observable("o", "Margin")`,
  which used to be named `"Margin[o]"`.
- BREAKING CHANGE: the ALL_CAPS class attributes and module-level constants
  are named in lower snake case, as in GEMSEO 7:
    - `BaseNoiser.SHORT_NAME` → [short_name][gemseo_umdo.disciplines.base_noiser.BaseNoiser.short_name];
    - `gemseo_umdo.formulations.factory.UMDO_FORMULATION_FACTORY`
      → [umdo_formulation_factory][gemseo_umdo.formulations.factory.umdo_formulation_factory].
- BREAKING CHANGE: the enumeration `gemseo_umdo.formulations.DOE_ALGO_NAMES`
  is renamed [DOEAlgorithmName][gemseo_umdo.formulations.DOEAlgorithmName].
- BREAKING CHANGE:
  [Sampling.callbacks][gemseo_umdo.formulations.sampling.Sampling.callbacks]
  only contains the callbacks added by the user;
  with `estimate_statistics_iteratively=True`,
  it no longer contains the iterative estimations of the statistics,
  which are passed to the DOE algorithm at each sampling
  and estimate both the statistics and,
  when the Jacobians of the outputs are evaluated, their Jacobians.
- [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  with `estimate_statistics_iteratively=False`
  reads the samples of the outputs and of their gradients at each design point
  in a single pass over the database of the evaluations of the disciplines,
  instead of one pass per output and per gradient,
  which reduces the cost of this option.
- BREAKING CHANGE:
  [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  raises a `ValueError` when estimating a variance, a standard deviation
  or a margin
  of an output that is available at a single sample of the uncertain variables,
  e.g. because the DOE algorithm generates a single sample,
  as [SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling]
  does at its first iteration with `initial_n_samples=1`,
  or because the evaluation fails at the other ones,
  or the Jacobian of such a statistic
  when the output and its gradient are both available at a single sample;
  previously,
  with `estimate_statistics_iteratively=True` (default),
  such a variance or standard deviation and its Jacobian were zero
  and such a margin and its Jacobian were those of the mean,
  and with `estimate_statistics_iteratively=False`,
  such a statistic was NaN and its Jacobian raised a `ZeroDivisionError`.
- BREAKING CHANGE:
  every U-MDO formulation,
  e.g. [Sampling][gemseo_umdo.formulations.sampling.Sampling],
  [TaylorPolynomial][gemseo_umdo.formulations.taylor_polynomial.TaylorPolynomial],
  [ControlVariate][gemseo_umdo.formulations.control_variate.ControlVariate],
  [Surrogate][gemseo_umdo.formulations.surrogate.Surrogate]
  and [PCE][gemseo_umdo.formulations.pce.PCE],
  raises a `ValueError`
  when adding an objective, a constraint or an observable
  whose statistic is `"Margin"` with a zero `factor` parameter,
  as such a margin is the mean;
  use the statistic `"Mean"` instead.
- BREAKING CHANGE:
  the default dispersion $\delta$ of an uncertain variable of
  [AdvancedBeamUncertainSpace][gemseo_umdo.use_cases.beam_model.advanced_uncertain_space.AdvancedBeamUncertainSpace]
  is 5% of its nominal value $\mu$ for the positive variables,
  1 degree for the angles `alpha` and `beta`,
  instead of about 57 degrees,
  and 5% of the nominal values of `b` and `h`
  for the coordinates `dy` and `dz` respectively,
  instead of 1.0 for all the variables,
  which gave very uneven levels of uncertainty,
  e.g. an almost deterministic Young modulus `E`
  and a thickness `t` varying by ±40%.
  A non-positive nominal value of a positive variable raises a `ValueError`.
- BREAKING CHANGE:
  [BaseUMDOFormulation.input_data_to_output_data][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation.input_data_to_output_data]
  maps each input point to a tuple of the form
  `(data_for_statistics, data_for_jacobians)` instead of a dictionary
  whose keys are `False` and `True`.
- Requesting the Jacobian of a statistic that the U-MDO formulation cannot estimate,
  e.g. `"Probability"` with [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  or any statistic with [Surrogate][gemseo_umdo.formulations.surrogate.Surrogate],
  raises a `NotImplementedError` naming the statistic, the output and the formulation,
  before sampling the disciplines.

### Removed

- Support for Python 3.10.
- BREAKING CHANGE: `UDOEScenario` was removed, as GEMSEO 7 removed `DOEScenario`;
  use [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  with the settings of a DOE algorithm to sample a U-MDO problem,
  or [UEvaluationScenario][gemseo_umdo.scenarios.u_evaluation_scenario.UEvaluationScenario]
  to evaluate statistics without objective or constraints.
- BREAKING CHANGE: `SobolGraph.DEFAULT_FILE_PATH`, which had no effect.
- BREAKING CHANGE: `Sampling.jacobian_callbacks`;
  use [Sampling.callbacks][gemseo_umdo.formulations.sampling.Sampling.callbacks],
  which are also called when estimating the Jacobians of the statistics.
- BREAKING CHANGE: `Sampling.input_data_to_output_samples`
  and `Surrogate.input_data_to_output_samples`, which were not used.

### Fixed

- The sampling-based U-MDO formulations sample the disciplines once per design point
  to estimate the objective and the constraints,
  instead of 1 + number of constraints times
  for [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  with `estimate_statistics_iteratively=True` (default)
  and twice for the other ones.
  When the driver requires the gradients and they are not approximated,
  this sampling is also used to estimate the gradients of all the statistics,
  instead of a second one;
  the disciplines are then linearized at every design point,
  including those where only the statistics are required,
  e.g. the line search points of a gradient-based optimizer
  (see
  [BaseUMDOFormulation.estimate_statistic_jacobians][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation.estimate_statistic_jacobians]).
  The results are the same, except for
  [SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling],
  which now increases the number of samples once per design point
  instead of once per statistic
  and uses $N_k$ samples at the $k$-th iteration
  for all the statistics and their gradients,
  as documented.
- [PCE][gemseo_umdo.formulations.pce.PCE] fits the polynomial chaos expansion
  once per design point to estimate the statistics and their gradients,
  instead of once per statistic and once more per gradient.
- [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  with `estimate_statistics_iteratively=True` (default)
  estimates correctly two statistics of the same output and of the same type
  with different parameters,
  e.g. two ``"Margin"`` with different factors;
  previously, both returned the value of the last one.
- [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  with `estimate_statistics_iteratively=True` (default)
  no longer lists `"BaseCentralMoment"` among its available statistics
  and no longer accepts it as a statistic name.
- [AdvancedBeamUncertainSpace][gemseo_umdo.use_cases.beam_model.advanced_uncertain_space.AdvancedBeamUncertainSpace]
  uses the standard deviation $\sigma=\delta/3$
  for a truncated normal variable whose dispersion $\delta$ is passed by the user,
  as documented;
  previously, it used $\sigma=\delta$,
  making the distribution 3 times too wide;
  its support is now $[\mu-\delta,\mu+\delta]$
  instead of $[\mu-3\delta,\mu+3\delta]$.
- [AdvancedBeamUncertainSpace][gemseo_umdo.use_cases.beam_model.advanced_uncertain_space.AdvancedBeamUncertainSpace]
  centers the uniform distributions of `b`, `h`, `t` and `L`
  on the `nominal_values` passed by the user;
  previously, it ignored them and used the default values.
- [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  with `estimate_statistics_iteratively=False`
  estimates the statistics of an output
  from the samples at which this output and, if required, its gradient were computed,
  discarding those at which the evaluation failed before,
  e.g. at the evaluation of a gradient;
  previously,
  the samples of these outputs were no longer aligned
  with those of the other outputs and of the gradients.
  For each sampling of the uncertain variables,
  it logs a single warning listing the outputs
  that are missing at some samples generated by the DOE algorithm,
  e.g. because their evaluation failed,
  with the numbers of samples used to estimate their statistics
  (and their Jacobians, from the samples where the gradients are also available),
  and those whose gradient is missing at all of them,
  and raises a `ValueError` when estimating a statistic of an output
  that is not available at any of them,
  or the Jacobian of a statistic of an output
  whose gradient is not available at any of them,
  e.g. because the DOE algorithm does not store the Jacobian.
- [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  with `estimate_statistics_iteratively=False`
  uses a sample generated several times by the DOE algorithm,
  e.g. by a custom DOE,
  as many times as it is generated;
  previously, it was used once, which biased the statistics.
- [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  with `estimate_statistics_iteratively=True` (default)
  estimates the Jacobian of a statistic at a design point
  where the statistic has not been estimated yet;
  previously, it raised an `AttributeError`.
  It raises a `ValueError` when estimating a statistic of an output
  that is not available at any sample of the uncertain variables,
  e.g. because its evaluation failed;
  previously, it raised an `AttributeError`.
  An error raised while sampling the uncertain variables at a design point
  no longer alters the statistics estimated at the next design points.
  As with `estimate_statistics_iteratively=False`,
  it logs a single warning per sampling of the uncertain variables
  listing the outputs that are missing at some samples
  generated by the DOE algorithm,
  with the numbers of samples used to estimate their statistics
  or their Jacobians;
  previously, these samples were skipped silently.
  Unlike with `estimate_statistics_iteratively=False`,
  a sample at which the evaluation of an output or of a gradient raises an error
  is not used for any output,
  as the DOE algorithm skips it,
  whereas the non-iterative estimation uses the outputs and gradients
  evaluated before the error;
  in particular,
  when the gradient of an output fails at all the samples,
  the `ValueError` points to the output instead of its gradient.
- [MLMCPilotFactory][gemseo_umdo.statistics.multilevel.mlmc.pilots.factory.MLMCPilotFactory]
  searches the pilots in the package
  `gemseo_umdo.statistics.multilevel.mlmc.pilots`;
  previously, it searched a non-existent package
  and reported it in its failed imports.
- When adding the objective, a constraint or an observable fails,
  e.g. because of invalid statistic parameters
  as in `scenario.add_observable("o", "Mean", foo=1)`
  or because its name is already used by another function,
  the U-MDO formulations are left unchanged,
  as when adding an observable already observed;
  previously, the output was still added as an observable
  of the problem of their MDO formulation.
- The U-MDO formulations reuse the data computed at a design point
  even if the caller modifies the array of this design point in place
  after the evaluation;
  previously, they sampled the disciplines again.
- [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  no longer changes the `callbacks`, `eval_jac` and `evaluate_observable_jacobian`
  of the settings of its DOE algorithm,
  even when a sampling fails;
  previously, a sampling left `eval_jac` and `evaluate_observable_jacobian`
  set to `True` in these settings when it estimated the gradients
  and a failed sampling left its callbacks in them.

## Version 4.3.0 (October 2025)

### Added

- The U-MDO formulation [PCE][gemseo_umdo.formulations.pce.PCE] can support any type of
  [BaseFCERegressor][gemseo.machine_learning.regression.core.base_fce.BaseFCERegressor],
  e.g., [PCERegressor][gemseo.machine_learning.regression.model.pce.PCERegressor] (default)
  and [FCERegressor][gemseo.machine_learning.regression.model.fce.FCERegressor].
- Support for Python 3.13.

### Removed

- Support for Python 3.9.

## Version 4.2.0 (August 2025)

### Added

- The [ControlVariate][gemseo_umdo.formulations.control_variate.ControlVariate] U-MDO formulation can create the control variates from any [BaseRegressor][gemseo.machine_learning.regression.core.base_regressor.BaseRegressor] using the arguments `regressor_settings` (for defining the regressor) and `regressor_doe_algo_settings` (for creating the training dataset). By default, this U-MDO formulation uses control variates based on Taylor polynomials.
- The subpackage [truss][gemseo_umdo.use_cases.truss] includes modules for implementing a truss structure problem from the literature.
- The function [create_noising_discipline_chain][gemseo_umdo.disciplines.utils.create_noising_discipline_chain] returns a disciplines chain to noise input variables.
  This function is used by the `UDOEScenario` and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  when the argument ``uncertain_design_variables`` is set to noise design variables.
  In the case of bi-level formulations,
  it may be preferable to use this function directly rather than through this argument,
  as illustrated in an example of the documentation.
- The argument `n_samples_increment` of
  the [SequentialSampling][gemseo_umdo.formulations.sampling.Sampling] U-MDO formulation
  can be either the increment of the sampling size
  or a function computing this increment from the current sampling size.
- The U-MDO formulations [Surrogate][gemseo_umdo.formulations.surrogate.Surrogate]
  and [PCE][gemseo_umdo.formulations.pce.PCE]
  store the quality of the surrogate models in the database attached to the scenario
  (see `scenario.formulation.optimization_problem.database`).

### Fixed

- The iterative Monte Carlo estimation of statistics when used for derivatives.

## Version 4.1.0 (April 2025)

### Added

- A [SobolGraph][gemseo_umdo.visualizations.sobol_graph.SobolGraph] can be defined
  from a [PCERegressor][gemseo.machine_learning.regression.model.pce.PCERegressor]
  by using its [from_pce][gemseo_umdo.visualizations.sobol_graph.SobolGraph.from_pce] method.

### Fixed

- The [PCE][gemseo_umdo.formulations.pce.PCE] U-MDO formulation handles the case
  where the uncertain space dimension is greater than 1
  and the derivatives with respect to the design variables are requested.
- The [Sampling][gemseo_umdo.formulations.sampling.Sampling] U-MDO formulation handles the case
  where the estimation of the standard deviation is zero
  and the derivatives with respect to the design variables are requested.

## Version 4.0.0 (March 2025)

### Added

- Each [BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
  has a Pydantic model to define its settings.
  For example,
  [Sampling_Settings][gemseo_umdo.formulations.sampling_settings.Sampling_Settings]
  is the Pydantic model for the [Sampling][gemseo_umdo.formulations.sampling.Sampling] U-MDO formulation.
- The U-MDO formulations
  [Sampling][gemseo_umdo.formulations.sampling.Sampling],
  [SequentialSampling][gemseo_umdo.formulations.sampling.Sampling]
  and [PCE][gemseo_umdo.formulations.pce.PCE]
  are differentiable,
  so that gradient-based optimizer can be used
  when the multidisciplinary process is differentiable.
- The U-MDO formulation [PCE][gemseo_umdo.formulations.pce.PCE] has a new option `approximate_statistics_jacobians`
  to approximate the Jacobians of the mean, standard deviation and variance with respect to the design variables
  at no extra cost,
  if you do not want to compute the derivatives of the disciplines to reduce the calculation budget
  or approximate these Jacobians by finite differences.
  The approximation uses the technique proposed by Mura _et al._ (2020)
  and is parametrized by the option `differentiation_step` (default: `1e-6`).
- [UOptAsUMDOScenario][gemseo_umdo.problems.uopt_as_umdo_scenario.UOptAsUMDOScenario]
  can make a monodisciplinary optimization problem under uncertainty multidisciplinary.
- An example illustrates the use of the [BiLevel][gemseo.formulation.bilevel.BiLevel] MDO formulation
  in the U-MDO formulation [Sampling][gemseo_umdo.formulations.sampling.Sampling].

### Changed

- API CHANGE:
  `UDOEScenario`
  and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  no longer have a default statistic estimation technique.
  The `statistic_estimation_settings` argument must be defined.
- API CHANGE:
  The `statistic_estimation` and `statistic_estimation_parameters` keyword arguments of
  `UDOEScenario`
  and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  have been replaced by the positional argument `statistic_estimation_settings`,
  which is a Pydantic model.
- API CHANGE:
  The statistic estimation settings of
  a [BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
  passed as positional and keyword arguments
  have been replaced by the unique positional argument `settings_model`,
  which is a Pydantic model.

## Fixed

- The bug related to the way
  [ControlVariate][gemseo_umdo.formulations.control_variate.ControlVariate]
  and [TaylorPolynomial][gemseo_umdo.formulations.taylor_polynomial.TaylorPolynomial]
  use non-normalized data has been corrected.
- `UDOEScenario`
  and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  now accept disciplines with variables that are floats rather than NumPy arrays.

## Version 3.0.0 (November 2024)

### Added

- Support GEMSEO v6.
- Support for Python 3.12.
- The U-MDO formulation [PCE][gemseo_umdo.formulations.pce.PCE] creates a polynomial chaos expansion (PCE)
  over the uncertain space at each iteration of the optimization loop and uses the coefficients of the PCE
  to estimate the following statistics: `Mean`, `StandardDeviation`, `Margin` and `Variance`.
- The U-MDO formulation [Surrogate][gemseo_umdo.formulations.surrogate.Surrogate] creates a surrogate model
  over the uncertain space at each iteration of the optimization loop and uses Monte Carlo sampling
  to estimate the following statistics: `Mean`, `StandardDeviation`, `Margin`, `Probability` and `Variance`.
- The U-MDO formulations
  [Sampling][gemseo_umdo.formulations.sampling.Sampling] and
  [SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling]
  have an option `samples_directory_path`
  to save the samples at each iteration of the algorithm chosen for the execution of the
  `UDOEScenario`
  or [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario].
- The U-MDO formulation [SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling]
  has an option `estimate_statistics_iteratively` (default: `True`)
  to compute the statistics iteratively
  and so do not store the samples in a `Database`.
- The dictionary argument `uncertain_design_variables` of
  `UDOEScenario`
  and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  can now accept values such as `("+", "u")` and `("*", "u")`
  to noise the corresponding key `x` as `x = dv_x + u` and `x = dv_x * (1 + u)`
  where `x` is a discipline input made uncertain by the random variable `u`.
- [AdditiveNoiser][gemseo_umdo.disciplines.additive_noiser.AdditiveNoiser]
  and
  [MultiplicativeNoiser][gemseo_umdo.disciplines.multiplicative_noiser.MultiplicativeNoiser]
  are disciplines to noise a design variable $x$ as $X=x+U$ and $X=x(1+U)$ respectively
  where $U$ is a random variable.
  [BaseNoiser][gemseo_umdo.disciplines.base_noiser.BaseNoiser]
  can be used to create other noising disciplines
  and a specific
  [NoiserFactory][gemseo_umdo.disciplines.noiser_factory.NoiserFactory]
  is available.
- [ControlVariate][gemseo_umdo.formulations.control_variate.ControlVariate],
  a new [BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
  estimating the statistics with a control variate technique based on Taylor polynomials.

### Changed

- The default value of the `initial_n_samples` argument of
  [SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling]
  is 2 instead of 1,
  because the default DOE algorithm (`"OT_OPT_LHS"`) requires at least 2 samples.
- `gemseo_umdo.scenarios._uscenario._UScenario` renamed to `gemseo_umdo.scenarios.base_u_scenario.BaseUScenario`.
- API CHANGE: `gemseo_umdo.statistics.mlmc.pilots.pilot.MLMCPilot` renamed to `gemseo_umdo.statistics.mlmc.pilots.base_mlmc_pilot.BaseMLMCPilot`.
- API CHANGE: `gemseo_umdo.statistics.mlmc_mlcv.pilots.pilot.MLMCMLCVPilot` renamed to `gemseo_umdo.statistics.mlmc_mlcv.pilots.base_mlmc_mlcv_pilot.BaseMLMCMLCVPilot`.
- API CHANGE: `gemseo_umdo.statistics.pilot.Pilot` renamed to `gemseo_umdo.statistics.base_pilot.BasePilot`.
- API CHANGE: `gemseo_umdo.formulations.formulation.UMDOFormulation` renamed to `gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation`.
- API CHANGE: `gemseo_umdo.formulations.statistics` is now a protected package.
- API CHANGE: `gemseo_umdo.formulations.functions` is now a protected package.
- The [BeamConstraints][gemseo_umdo.use_cases.beam_model.constraints.BeamConstraints] discipline
  computed outputs of the form `a/(b+eps)` where `eps` was used to avoid division by zero.
  Now,
  this discipline computes outputs of the form `b/a`
  as `a` is never zero.

### Fixed

- The U-MDO formulation [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  works properly when the option `estimate_statistics_iteratively` is `True`
  and the DOE option `n_processes` is greater than 1.
- The docstring of the `uncertain_design_variables` argument of
  `UDOEScenario`
  and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  explains that specifying a value such as `"{} + u"` at key `"x"`
  assumes that both the uncertain design variable `"x"`
  and the uncertain variable `"u"` are scalar variables.
- The discipline transforming the design variables into uncertain design variables
  is placed before the user's disciplines;
  by doing so,
  the uncertain design variables can be propagated
  through the multidisciplinary process
  even with MDO formulations that do not ensure the satisfaction of couplings,
  such as [DisciplinaryOpt][gemseo.formulation.disciplinary_opt.DisciplinaryOpt].

## Version 2.0.1 (January 2024)

### Fixed

- The U-MDO formulations handle the finite-difference approximation of derivatives.

## Version 2.0.0 (December 2023)

### Added

- Support for Python 3.11.
- A web documentation.
- The heat equation problem (
  [HeatEquationConfiguration][gemseo_umdo.use_cases.heat_equation.configuration.HeatEquationConfiguration],
  [HeatEquationDiscipline][gemseo_umdo.use_cases.heat_equation.discipline.HeatEquation],
  [HeatEquationModel][gemseo_umdo.use_cases.heat_equation.model.HeatEquationModel]
  and [HeatEquationUncertainSpace][gemseo_umdo.use_cases.heat_equation.uncertain_space.HeatEquationUncertainSpace])
  to illustrate the algorithms [MLMC][gemseo_umdo.statistics.multilevel.mlmc.mlmc.MLMC]
  and [MLMCMLCV][gemseo_umdo.statistics.multilevel.mlmc_mlcv.mlmc_mlcv.MLMCMLCV].
- The [MLMC][gemseo_umdo.statistics.multilevel.mlmc.mlmc.MLMC] and
  the [MLMCMLCV][gemseo_umdo.statistics.multilevel.mlmc_mlcv.mlmc_mlcv.MLMCMLCV]
  algorithms to estimate a statistic of the output of a function
  whose input is random.
- The [MonteCarloSampler][gemseo_umdo.monte_carlo_sampler.MonteCarloSampler]
  to sample vectorized functions.
- [UncertainCouplingGraph][gemseo_umdo.visualizations.uncertain_coupling_graph.UncertainCouplingGraph]
  has a new option `save` (default: `True`).
- The U-MDO formulation [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  has an option `estimate_statistics_iteratively` (default: `True`)
  to compute the statistics iteratively
  and so do not store the samples in a `Database`.
- The package `gemseo_umdo.formulations.functions` contains the `MDOFunction`s
  used by a [UMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
  to compute the statistics of the objective, constraints and observables.
- The logs of the `UDOEScenario`
  and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  include the uncertain space.

### Changed

- Setting the argument `n_samples`
  of the U-MDO formulation [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  is mandatory for many DOE algorithms
  but optional in the case where
  the DOE algorithm does not consider a `n_samples` argument to generate the samples.
- The estimator of the `Variance`
  used by the U-MDO formulation [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  with `estimate_statistics_iteratively=False` is now unbiased.
- API changes:
    - The options of the statistics estimators
    are now set at instantiation instead of execution.
    - `gemseo_umdo.estimators` has been renamed to `gemseo_umdo.formulations.statistics`.
- The log of the statistics no longer includes design variables and uncertain inputs
  (e.g. `E[y(x; u)]`),
  but only uncertain output (e.g. `E[y]`) to avoid display problems in large dimensions.

### Fixed

- The `UDOEScenario`
  and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  maximize the statistic of the objective
  when the argument `maximize_objective` is set to `True`.
- The log of the objective and constraint is now consistent
  with the arguments `maximize_objective` and `constraint_name`.

### Removed

- Support for Python 3.8.

## Version 1.1.1 (October 2023)

### Fixed

- One test was not compatible with GEMSEO 5.1+.

## Version 1.1.0 (June 2023)

### Added

- The beam problem ([Beam][gemseo_umdo.use_cases.beam_model.discipline.Beam],
  [BeamConstraints][gemseo_umdo.use_cases.beam_model.constraints.BeamConstraints],
  [BeamUncertainSpace][gemseo_umdo.use_cases.beam_model.uncertain_space.BeamUncertainSpace]
  and [BeamDesignSpace][gemseo_umdo.use_cases.beam_model.design_space.BeamDesignSpace]
  to benchmark robust optimization algorithms.
- [TaylorPolynomial][gemseo_umdo.formulations.taylor_polynomial.TaylorPolynomial],
  a new [UMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
  estimating the statistics with Taylor polynomials.
- [SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling],
  a new [UMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
  estimating the statistics with sequential sampling.
- [UncertainCouplingGraph][gemseo_umdo.visualizations.uncertain_coupling_graph.UncertainCouplingGraph]
  to visualize the dispersion of the coupling variables.
- [SobolGraph][gemseo_umdo.visualizations.sobol_graph.SobolGraph]
  to visualize the first-, second- and total-order Sobol' indices.
- The set of [SpringMassModel][gemseo_umdo.use_cases.spring_mass_model.model.SpringMassModel],
  [SpringMassDiscipline][gemseo_umdo.use_cases.spring_mass_model.discipline.SpringMassDiscipline]
  and [SpringMassUncertainSpace][gemseo_umdo.use_cases.spring_mass_model.uncertain_space.SpringMassUncertainSpace]
  is a use case based on a spring-mass system.

### Fixed

- The `_UScenario` no longer changes the list of disciplines passed by the user.

## Version 1.0.1 (January 2023)

### Changed

- API change: the argument `statistic_estimation_options`
  of [UMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
  has been renamed to `statistic_estimation_parameters`.
- API change: `UMDOFormulation._processed_functions` replaces `Sampling.processed_functions`.

## Version 1.0.0 (July 2022)

First release.
