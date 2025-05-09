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

- The subpackage [truss][gemseo_umdo.use_cases.truss] includes modules for implementing a truss structure problem from the literature.
- The function [create_noising_discipline_chain][gemseo_umdo.disciplines.utils.create_noising_discipline_chain] returns a disciplines chain to noise input variables.
  This function is used by the [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario] and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  when the argument ``uncertain_design_variables`` is set to noise design variables.
  In the case of bi-level formulations,
  it may be preferable to use this function directly rather than through this argument,
  as illustrated in an example of the documentation.

### Fixed

- The iterative Monte Carlo estimation of statistics when used for derivatives.

## Version 4.1.0 (April 2025)

### Added

- A [SobolGraph][gemseo_umdo.visualizations.sobol_graph.SobolGraph] can be defined
  from a [PCERegressor][gemseo.mlearning.regression.algos.pce.PCERegressor]
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
- An example illustrates the use of the [BiLevel][gemseo.formulations.bilevel.BiLevel] MDO formulation
  in the U-MDO formulation [Sampling][gemseo_umdo.formulations.sampling.Sampling].

### Changed

- API CHANGE:
  [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
  and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  no longer have a default statistic estimation technique.
  The `statistic_estimation_settings` argument must be defined.
- API CHANGE:
  The `statistic_estimation` and `statistic_estimation_parameters` keyword arguments of
  [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
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
- [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
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
  [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
  or [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario].
- The U-MDO formulation [SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling]
  has an option `estimate_statistics_iteratively` (default: `True`)
  to compute the statistics iteratively
  and so do not store the samples in a `Database`.
- The dictionary argument `uncertain_design_variables` of
  [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
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
  works properly when the option `estimate_statistics_iteratively`  is `True`
  and the DOE option `n_processes` is greater than 1.
- The docstring of the `uncertain_design_variables` argument of
  [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
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
  such as [DisciplinaryOpt][gemseo.formulations.disciplinary_opt.DisciplinaryOpt].

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
- The logs of the [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
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
  but only uncertain output  (e.g. `E[y]`) to avoid display problems in large dimensions.

### Fixed

- The [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
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
