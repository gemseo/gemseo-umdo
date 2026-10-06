<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
# MDO under uncertainty

This section describes the design of
the [disciplines][gemseo_umdo.disciplines],
[formulations][gemseo_umdo.formulations]
and [scenarios][gemseo_umdo.scenarios]
subpackages
used to define and solve an MDO problem under uncertainty.

!!! info

    Open the [user guide](../user_guide/umdo/index.md) for general information, e.g. concepts, API, examples, etc.

## Tree structure

```tree
gemseo_umdo
  disciplines # Subpackage including noising disciplines
    additive_noiser.py # Noising discipline adding a random variable to a deterministic one
    base_noiser.py # Base class for noising disciplines
    multiplicative_noiser.py # Noising discipline multiplying a deterministic variable by a random one
    noiser_factory.py # Factory of noising disciplines
    utils.py # Function creating the chain of noising disciplines
  formulations # Subpackage including U-MDO formulations
    factory.py # Factory of U-MDO formulations
    base_umdo_formulation.py # Base class for U-MDO formulations
    base_umdo_formulation_settings.py # Base class for the settings of the U-MDO formulations
    base_sampling_settings.py # Base class for the settings of the sampling-based U-MDO formulations
    base_surrogate_settings.py # Base classes for the settings of the surrogate-based U-MDO formulations
    control_variate.py # U-MDO formulation estimating statistics using Taylor-based control variates
    control_variate_settings.py # Settings for ControlVariate
    pce.py # U-MDO formulation estimating statistics using polynomial chaos expansions (PCE)
    pce_settings.py # Settings for PCE
    sampling.py # U-MDO formulation estimating statistics using Monte Carlo sampling
    sampling_settings.py # Settings for Sampling
    sequential_sampling.py # U-MDO formulation estimating statistics using sequential Monte Carlo sampling
    sequential_sampling_settings.py # Settings for SequentialSampling
    surrogate.py # U-MDO formulation estimating statistics using Monte Carlo sampling of a surrogate model
    surrogate_settings.py # Settings for Surrogate
    taylor_polynomial.py # U-MDO formulation estimating statistics using Taylor polynomials
    taylor_polynomial_settings.py # Settings for TaylorPolynomial
    _functions # Subpackage of statistic estimation functions to be used with EvaluationProblem
      base_statistic_function.py # Base class for statistic estimation functions
      statistic_function_for_a_specific_u_mdo_formulation.py # Statistic estimation functions for a U-MDO formulation
      ...
    _statistics # Subpackage of statistic estimators
      base_statistic_estimator.py # Base class for statistic estimators
      specific_u_mdo_formulation # The subpackage of statistic estimators associated with a specific U-MDO formulation
        base_sampling_estimator.py # The base class for statistic estimators associated with this U-MDO formulation
        mean.py # The estimator of the mean associated with this U-MDO formulation
        variance.py # The estimator of the variance associated with this U-MDO formulation
        ...
      ...
  scenarios # Subpackage including scenarios using U-MDO formulations
    base_u_scenario.py # Base scenario using a U-MDO formulation
    udoe_scenario.py # DOE-based scenario using a U-MDO formulation
    umdo_scenario.py # Optimizer-based scenario using a U-MDO formulation
```

## Class diagram

A `BaseUScenario` is a mixin
adapting the API of an `MDOScenario`
to the definition of the uncertain space, statistics and the associated estimation techniques;
`UMDOScenario` and `UDOEScenario` derive from both `BaseUScenario` and `MDOScenario`.

As any `MDOScenario`,
a `BaseUScenario` creates an `OptimizationProblem` over the design space
and passes it to its formulation,
which is a `BaseUMDOFormulation`.
The objective, constraints and observables are then added
with the methods `add_objective()`, `add_constraint()` and `add_observable()`,
which take the name of the statistic to be applied to the outputs.
When uncertain design variables are defined,
the `BaseUScenario` also prepends a chain of `BaseNoiser` disciplines to the disciplines.

A `BaseUMDOFormulation` is a `BaseFormulation` made of

- the settings of a specific statistics estimation technique,
  i.e. a `BaseUMDOFormulationSettings`, e.g. `Sampling_Settings` for sampling,
- a standard `BaseMDOFormulation`, e.g. `MDF`,
  created from its settings (`MDF_Settings` by default)
  over an `EvaluationProblem` defined over the uncertain space,
  i.e. a `RandomSpace`;
  the outputs of the objective, constraints and observables
  are observables of this `EvaluationProblem`,
- optionally, an auxiliary `BaseMDOFormulation`
  whose functions are differentiable with respect to the uncertain variables
  (see `_USE_AUXILIARY_MDO_FORMULATION`).

The standard `BaseMDOFormulation` is in charge to define the multidisciplinary process
for a specific design value and a specific uncertainty value
while the estimation technique is in charge to

1. sample this multidisciplinary process over the uncertain space,
2. estimate the statistics by means of `BaseStatisticFunction`s
   which are particular `ArrayFunction`s
   attached to the `OptimizationProblem` of the `BaseUMDOFormulation`.

A `BaseStatisticFunction` relies on a basic functor, called `BaseStatisticEstimator`.

So,
adding a new U-MDO formulation `Foo` implies to

- subclass `BaseUMDOFormulation` to `Foo`,
  whose constructor has the signature
  `(problem, disciplines, settings=None, *, uncertain_space, mdo_formulation_settings=None)`,
- subclass `BaseUMDOFormulationSettings` to `Foo_Settings`
  (the naming convention `Foo_Settings` binds the settings to `Foo`),
- subclass `BaseStatisticFunction` to `StatisticFunctionForFoo`,
- subclass `BaseStatisticEstimator` to `BaseFooEstimator`,
- subclass `BaseFooEstimator` to `Mean`, `Variance`, etc.

``` mermaid
classDiagram

   BaseUScenario <|-- UMDOScenario
   MDOScenario <|-- UMDOScenario
   BaseUScenario <|-- UDOEScenario
   MDOScenario <|-- UDOEScenario

   class BaseUScenario {
    +add_objective()
    +add_constraint()
    +add_observable()
    +available_statistics
    +formulation_name
    +mdo_formulation
    +uncertain_space
   }

   BaseUScenario *-- BaseUMDOFormulation
   BaseUScenario "1" *-- "n" BaseNoiser
   BaseNoiser --|> Discipline
   BaseFormulation <|-- BaseUMDOFormulation
   BaseFormulation <|-- BaseMDOFormulation
   BaseUMDOFormulation o-- BaseMDOFormulation: MDO formulation(s)

   class BaseUMDOFormulation {
     +add_constraint()
     +add_observable()
     +auxiliary_mdo_formulation
     +available_statistics
     +create_constraint()
     +create_objective()
     +get_top_level_disciplines()
     +input_data_to_output_data
     +mdo_formulation
     +name
     +uncertain_space
     +update_top_level_disciplines()
   }

   BaseUMDOFormulation *-- OptimizationProblem: over the design space
   OptimizationProblem "1" o-- "n" BaseStatisticFunction
   BaseMDOFormulation *-- EvaluationProblem: over the uncertain space
   EvaluationProblem o-- RandomSpace
   BaseUMDOFormulation o-- RandomSpace: uncertain space
   BaseUMDOFormulation "1" --> "n" BaseStatisticFunction
   BaseStatisticFunction *-- BaseStatisticEstimator

   BaseUMDOFormulation <|-- Sampling
   ArrayFunction <|-- BaseStatisticFunction
   BaseStatisticFunction <|-- StatisticFunctionForStandardSampling
   Sampling "1" --> "n" StatisticFunctionForStandardSampling
   StatisticFunctionForStandardSampling *-- BaseSamplingEstimator
   BaseStatisticEstimator <|-- BaseSamplingEstimator
   BaseSamplingEstimator <|-- Mean

   BaseUMDOFormulation *-- BaseUMDOFormulationSettings
   BaseFormulationSettings <|-- BaseUMDOFormulationSettings

   <<mixin>> BaseUScenario
   <<abstract>> BaseFormulation
   <<abstract>> BaseMDOFormulation
   <<abstract>> BaseUMDOFormulation
   <<abstract>> BaseStatisticFunction
   <<abstract>> BaseStatisticEstimator
   <<abstract>> BaseSamplingEstimator
   <<abstract>> BaseNoiser

   namespace Example {
    class Sampling
    class StatisticFunctionForStandardSampling
    class BaseSamplingEstimator
    class Mean
   }

   namespace gemseo {
     class ArrayFunction
     class BaseFormulation
     class BaseFormulationSettings
     class BaseMDOFormulation
     class Discipline
     class EvaluationProblem
     class MDOScenario
     class OptimizationProblem
     class RandomSpace
   }
```
