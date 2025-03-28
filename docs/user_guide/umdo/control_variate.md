<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Control variate

[ControlVariate][gemseo_umdo.formulations.control_variate.ControlVariate]
is a U-MDO formulation that estimates the statistics
using control variates based on first-order Taylor polynomials.

!!! note "Control variates (CVs) method"

    The control variates method is a variance reduction technique used in Monte Carlo sampling.
    [Read more](https://en.wikipedia.org/wiki/Control_variates)

The Taylor polynomials are centered at $\mu=\mathbb{E}[U]$
where $U$ is the random input vector.

This U-MDO formulation has one mandatory parameter, namely `n_samples`.

Here is a typical scenario template:

``` py
scenario = UMDOScenario(
    disciplines,
    mdo_formulation_name,
    objective_name,
    design_space,
    uncertain_space,
    statistic_name,
    statistic_estimation_settings=ControlVariate_Settings(n_samples=20),
)
```

## Settings

By default,
the formulation uses the DOE algorithm `OT_OPT_LHS` with 10 samples:
the Latin hypercube sampling (LHS)
enhanced by simulated annealing
of OpenTURNS.
Simulated annealing is a global optimization technique that
starts from an initial LHS
and improves it to maximize its discrepancy
and so to get a better space-filling LHS.

The number of samples can be changed with the parameter `n_samples`
and the DOE algorithm name can be changed with the parameter `doe_algo_settings` ,
which is a Pydantic model deriving from [BaseDOESettings][gemseo.algos.doe.base_doe_settings.BaseDOESettings].
When `n_samples` is `None` (default) and `doe_algo_settings` has a field `n_samples`,
then this field is considered.
When `doe_algo_settings` has a field `seed` and its value is `None`,
then the U-MDO formulation will use [SEED][gemseo.utils.seeder.SEED].

!!! note "API"
    Use `statistic_estimation_settings`
    to set the algorithm name and settings,
    _e.g._

    ``` py
    settings = ControlVariate_Settings(doe_algo_settings=OT_MONTE_CARLO(n_samples=20, n_processes=2))
    scenario = UMDOScenario(
        disciplines,
        mdo_formulation_name,
        objective_name,
        design_space,
        uncertain_space,
        statistic_name,
        statistic_estimation_settings=settings,
    )
    ```

## Statistics

This U-MDO formulation has been implemented
for the expectation, the standard deviation, the variance and the margin.

Only the average formula is noted here, for simplicity's sake

$$\mathbb{E}[\varphi(x,U)]
\approx
\frac{1}{N}\sum_{i=1}^N f\left(x,U^{(i)}\right)
+\alpha_N\left(\frac{1}{N}\sum_{j=1}^N \tilde{f}\left(x,U^{(j)}\right)-f(x,\mu)\right)$$

where $\tilde{f}(x)$ is the first-order Taylor polynomial of $f(x)$ at $\mu$,
$\alpha_N$ is the empirical estimator
of $\frac{\text{cov}\left[f(x,U),\tilde{f}(x,u)\right]}
{\mathbb{V}\left[f(x,U)\right]}$
and $U^{(1)},\ldots,U^{(N)}$ are $N$ independent realizations of $U$.
