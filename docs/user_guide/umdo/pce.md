<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Polynomial chaos expansion

The U-MDO formulation [PCE][gemseo_umdo.formulations.pce.PCE]
can solve an MDO problem
associated with an [MDOFormulation][gemseo.formulations.mdo_formulation.MDOFormulation]
with polynomial chaos expansions (PCEs).

At each iteration of the optimization loop,
a PCE is built over the uncertain space
and its coefficients are used to estimate specific statistics,
namely mean, standard deviation, variance and margin.

The number of samples to build the PCE is mandatory
and must be set with the parameter `doe_n_samples`.

Here is a typical scenario template:

``` py
scenario = UMDOScenario(
    disciplines,
    mdo_formulation_name,
    objective_name,
    design_space,
    uncertain_space,
    statistic_name,
    statistic_estimation="PCE",
    statistic_estimation_parameters={"doe_n_samples": 20}
)
```

## Options

### DOE algorithm

By default,
the formulation uses the DOE algorithm `OT_OPT_LHS`:
the Latin hypercube sampling (LHS)
enhanced by simulated annealing
of OpenTURNS.
Simulated annealing is a global optimization technique that
starts from an initial LHS
and improves it to maximize its discrepancy
and so to get a better space-filling LHS.

The DOE algorithm can be set with the string parameter `doe_algo`
and its options with the dictionary parameter `doe_algo_options`.

!!! note "API"
    Use `statistic_estimation_parameters`
    to set the algorithm name and options,
    e.g.

    ``` py
    scenario = UMDOScenario(
        disciplines,
        mdo_formulation_name,
        objective_name,
        design_space,
        uncertain_space,
        statistic_name,
        statistic_estimation_parameters={
            "doe_algo": "OT_MONTE_CARLO",
            "doe_n_samples": 20,
            "doe_algo_options": {"n_processes": 2}
        }
    )
    ```

### PCE options

This U-MDO formulation is based on the [PCERegressor][gemseo.mlearning.regression.algos.pce.PCERegressor] available in GEMSEO,
which wraps the [OpenTURNS' PCE algorithm](https://openturns.github.io/openturns/latest/user_manual/response_surface/_generated/openturns.FunctionalChaosAlgorithm.html).
Use the `pce_options` arguments to set the options of the [PCERegressor][gemseo.mlearning.regression.algos.pce.PCERegressor].

## Statistics

This formulation has been implemented for the expectation and variance,
as well as combinations of these statistics,
from the coefficients $(\alpha_i)_{1\leq i \leq N\}$ of the PCE

$$\hat{f}_x(U)=\alpha_0 + \sum_{1<i\leq P}\alpha_i\Phi_i(U).$$

### Mean

$$\mathbb{E}[\varphi(x,U)] \approx \alpha_0$$

### Variance

$$\mathbb{V}[\varphi(x,U)] \approx \sum_{1<i\leq P}\alpha_i^2$$

### Standard deviation

$$\mathbb{S}[\varphi(x,U)] \approx \sqrt{\sum_{1<i\leq P}\alpha_i^2}$$

### Margin

$$\textrm{Margin}[\varphi(x,U)] \approx \alpha_0 + \kappa \times \sqrt{\sum_{1<i\leq P}\alpha_i^2}$$
