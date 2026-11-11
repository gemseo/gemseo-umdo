<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Sampling

[Sampling][gemseo_umdo.formulation.sampling.Sampling]
is a U-MDO formulation that estimates the statistics unbiasedly
using Monte Carlo sampling.

Here is a typical scenario template:

``` py
scenario = UMDOScenario(
    disciplines,
    design_space,
    uncertain_space,
    Sampling_Settings(n_samples=20),
    formulation_settings=mdo_formulation_settings,
)
scenario.add_objective(objective_name, statistic_settings)
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
and the DOE algorithm name can be changed with the parameter `doe_algo_settings`,
which is a Pydantic model deriving from [BaseDOESettings][gemseo.doe.core.base_doe_settings.BaseDOESettings].
When `n_samples` is `None` (default) and `doe_algo_settings` has a field `n_samples`,
then this field is considered.
When `doe_algo_settings` has a field `seed` and its value is `None`,
then the U-MDO formulation will use [seed][gemseo.util.seeder.seed].

!!! note "API"
    Use `statistic_estimation_settings`
    to set the DOE algorithm name and settings,
    e.g.

    ``` py
    settings = Sampling_Settings(
        doe_algo_settings=OT_MONTE_CARLO_Settings(n_samples=20, n_processes=2)
    )
    scenario = UMDOScenario(
        disciplines,
        design_space,
        uncertain_space,
        settings,
        formulation_settings=mdo_formulation_settings,
    )
    scenario.add_objective(objective_name, statistic_settings)
    ```

## Statistics

This U-MDO formulation has been implemented
for the expectation, the standard deviation, the variance, the margin, the probability
and the quantile.

| Statistic          | Notation                         | Estimator                                                                                        |
|--------------------|----------------------------------|--------------------------------------------------------------------------------------------------|
| Mean               | $\mathbb{E}[\varphi(x,U)]$       | $E_N[\varphi(x,U)]=\frac{1}{N}\sum_{i=1}^N\varphi(x,u^{(i)})$                                    |
| Variance           | $\mathbb{V}[\varphi(x,U)]$       | $V_N[\varphi(x,U)]=\frac{1}{N-1}\sum_{i=1}^N\left(\varphi(x,u^{(i)})-E_N[\varphi(x,U)]\right)^2$ |
| Standard deviation | $\mathbb{S}[\varphi(x,U)]$       | $S_N[\varphi(x,U)]=\sqrt{V_N[\varphi(x,U)]}$                                                     |
| Margin             | $\textrm{Margin}[\varphi(x,U)]$  | $\textrm{Margin}_N[\varphi(x,U)]=E_N[\varphi(x,U)]+\kappa\cdot S_N[\varphi(x,U)]$                |
| Probability        | $\mathbb{P}[\varphi(x,U)\leq 0]$ | $P_N[\varphi(x,U)\leq 0]=E_N[\mathbb{1}_{\varphi(x,U)\leq 0}]$                                   |
| Quantile           | $Q_\alpha[\varphi(x,U)]$         | $Q_{\alpha,N}[\varphi(x,U)]=(1-w)\varphi_{(j)}(x)+w\varphi_{(j+1)}(x)$                           |

where $\varphi_{(1)}(x)\leq\ldots\leq\varphi_{(N)}(x)$ are the order statistics
of $\varphi(x,u^{(1)}),\ldots,\varphi(x,u^{(N)})$
and where the index $j$ and the weight $w\in[0,1[$ depend on $\alpha$, $N$
and the method of [numpy.quantile][numpy.quantile] [@hyndman1996sample]
set by the field `method` of
[Quantile_Settings][gemseo_umdo.statistic.quantile_settings.Quantile_Settings]
(default: `"linear"`),
e.g. `scenario.add_constraint("c", Quantile_Settings(level=0.95, method="hazen"))`.
The default level $\alpha$ is 0.95.
A warning is logged when $N\min(\alpha,1-\alpha)<10$
as the estimation is then inaccurate.

!!! warning "Quantile and iterative estimation"

    The quantile cannot be estimated iteratively:
    using it requires `estimate_statistics_iteratively=False`,
    otherwise an error is raised.

## Gradient-based optimization

When the multidisciplinary process is differentiable,
and a gradient-based optimizer is used,
analytical derivatives are implemented with the following statistics:
mean, standard deviation, variance, margin and quantile.
For probability statistics,
only derivatives approximated by finite differences are currently available.

The derivative of the quantile estimator is
$(1-w)\nabla_x\varphi_{(j)}(x)+w\nabla_x\varphi_{(j+1)}(x)$,
which is the pathwise estimator of the quantile sensitivity [@hong2009estimating].
This estimator is only piecewise differentiable:
its gradient jumps when the ranking of the samples changes.

- What it guarantees:
  with a fixed sample,
  i.e. with common random numbers as by default,
  it is the exact gradient of the estimated quantile
  wherever the ranking of the samples does not change,
  so that the optimizer is consistent with the function it minimizes.
- What it does not guarantee:
  as it only uses the gradients at one or two samples,
  it is not a consistent estimator of the gradient of the quantile
  $\nabla_x Q_\alpha[\varphi(x,U)]=\mathbb{E}[\nabla_x\varphi(x,U)|\varphi(x,U)=Q_\alpha[\varphi(x,U)]]$:
  its variance does not decrease when $N$ increases.

Consistent estimators of this gradient exist,
e.g. by batching the samples [@hong2009estimating]
or with a kernel estimator [@liu2009kernel],
but they are not the gradient of the estimated quantile.
