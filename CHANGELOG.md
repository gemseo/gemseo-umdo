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
  has a new option ``save`` (default: ``True``).
- The U-MDO formulation [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  has an option ``estimate_statistics_iteratively`` (default: ``True``)
  to compute the statistics iteratively
  and so do not store the samples in a ``Database``.
- The package ``gemseo_umdo.formulations.functions`` contains the ``MDOFunction``s
  used by a [UMDOFormulation][gemseo_umdo.formulations.formulation.UMDOFormulation]
  to compute the statistics of the objective, constraints and observables.
- The logs of the [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
  and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  include the uncertain space.

### Changed

- Setting the argument ``n_samples``
  of the U-MDO formulation [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  is mandatory for many DOE algorithms
  but optional in the case where
  the DOE algorithm does not consider a ``n_samples`` argument to generate the samples.
- The estimator of the [Variance][gemseo_umdo.formulations.statistics.sampling.variance.Variance]
  used by the U-MDO formulation [Sampling][gemseo_umdo.formulations.sampling.Sampling]
  with ``estimate_statistics_iteratively=False`` is now unbiased.
- API changes:
  - The options of the statistics estimators
    are now set at instantiation instead of execution.
  - ``gemseo_umdo.estimators`` has been renamed to ``gemseo_umdo.formulations.statistics``.
- The log of the statistics no longer includes design variables and uncertain inputs
  (e.g. ``E[y(x; u)]``),
  but only uncertain output  (e.g. ``E[y]``) to avoid display problems in large dimensions.

### Fixed

- The [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
  and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  maximize the statistic of the objective
  when the argument ``maximize_objective`` is set to ``True``.
- The log of the objective and constraint is now consistent
  with the arguments ``maximize_objective`` and ``constraint_name``.

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
  a new [UMDOFormulation][gemseo_umdo.formulations.formulation.UMDOFormulation]
  estimating the statistics with Taylor polynomials.
- [SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling],
  a new [UMDOFormulation][gemseo_umdo.formulations.formulation.UMDOFormulation]
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

- The ``_UScenario`` no longer changes the list of disciplines passed by the user.

## Version 1.0.1 (January 2023)

### Changed

- API change: the argument ``statistic_estimation_options``
  of [UMDOFormulation][gemseo_umdo.formulations.formulation.UMDOFormulation]
  has been renamed to ``statistic_estimation_parameters``.
- API change: ``UMDOFormulation._processed_functions`` replaces ``Sampling.processed_functions``.

### Version 1.0.0 (July 2022)

First release.
