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

    Open the [user guide](../user_guide/umdo/index.md) for general information, *e.g.* concepts, API, examples, etc.

## Tree structure

```tree
gemseo_umdo
  disciplines # Subpackage including noising disciplines
    additive_noiser.py # Noising discipline adding a random variable to a deterministic one
    base_noiser.py # Base class for noising disciplines
    multiplicative_noiser.py # Noising discipline multiplying a deterministic variable by a random one
    noiser_factory.py # Factory of noising disciplines
  formulations # Subpackage including U-MDO formulations
    factory.py # Factory of U-MDO formulations
    base_umdo_formulation.py # Base class for U-MDO formulations
    control_variate.py # U-MDO formulation estimating statistics using Taylor-based control variates
    pce.py # U-MDO formulation estimating statistics using polynomial chaos expansions (PCE)
    sampling.py # U-MDO formulation estimating statistics using Monte Carlo sampling
    sequential_sampling.py # U-MDO formulation estimating statistics using Monte Carlo sampling
    taylor_polynomial.py # U-MDO formulation estimating statistics using Taylor polynomials
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

A `BaseUScenario` is a `Scenario`
with an API adapted to the definition of the uncertain space, statistics and the associated estimation techniques.

A `BaseUScenario` is made of

- a `BaseUMDOFormulation`, which is an `MDOFormulation` depending on a standard `MDOFormulation`, e.g. `MDF`,
- a specific statistics estimation technique,
  *e.g.* sampling,
  whose class name corresponds to this technique,
  e.g. `Sampling`.

The standard `MDOFormulation` is in charge to define the multidisciplinary process
for a specific design value and a specific uncertainty value
while the estimation technique is in charge to

1. sample this multidisciplinary process over the uncertain space,
2. estimate the statistics by means of `BaseStatisticFunction`s
   which are particular `MDOFunction`s
   associated with the `OptimizationProblem` attached to the `UMDOFormulation`.

A `BaseStatisticFunction` relies on a basic functor, called `BaseStatisticEstimator`.

So,
adding a new U-MDO formulation `Foo` implies to

- subclass `BaseUMDOFormulation` to `Foo`,
- subclass `BaseStatisticFunction` to `StasticFunctionForFoo`,
- subclass `BaseStatisticEstimator` to `BaseFooEstimator`,
- subclass `BaseFooEstimator` to `Mean`, `Variance`, etc.

``` mermaid
classDiagram

   BaseUScenario --|> Scenario
   BaseUScenario <|-- UMDOScenario
   MDOScenario <|-- UMDOScenario

   class BaseUScenario {
    +add_constraint()
    +add_observable()
    +formulation_name
    +mdo_formulation
    +uncertain_space
   }

   BaseUScenario *-- BaseUMDOFormulation
   BaseUMDOFormulation <|-- BaseMDOFormulation
   BaseUMDOFormulation o-- BaseMDOFormulation

   class BaseUMDOFormulation {
     +add_constraint()
     +add_observable()
     +get_expected_dataflow()
     +get_expected_workflow()
     +get_top_level_disc()
     +input_data_to_output_data
     +mdo_formulation
     +name
     +uncertain_space
     +update_top_level_disciplines()
   }

   BaseUMDOFormulation o-- ParameterSpace: uncertain space
   BaseUMDOFormulation "1" --> "n" BaseStatisticFunction
   BaseStatisticFunction *-- BaseStatisticEstimator

   BaseUMDOFormulation <|-- Sampling
   MDOFunction <|-- BaseStatisticFunction
   BaseStatisticFunction <|-- StatisticFunctionForStandardSampling
   Sampling "1" --> "n" StatisticFunctionForStandardSampling
   StatisticFunctionForStandardSampling *-- BaseSamplingEstimator
   BaseStatisticEstimator <|-- BaseSamplingEstimator
   BaseSamplingEstimator <|-- Mean

   BaseUMDOFormulation *-- OptimizationProblem
   OptimizationProblem "1" o-- "n" BaseStatisticFunction

   BaseUMDOFormulation "1" *-- "n" BaseNoiser
   BaseNoiser --|> MDODiscipline

   <<abstract>> BaseUScenario
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
     class MDODiscipline
     class MDOFunction
     class MDOScenario
     class ParameterSpace
     class Scenario
   }
```
