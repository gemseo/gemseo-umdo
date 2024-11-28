<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# MDO under uncertainty

## Introduction

### Optimization problem

A standard optimization problem aims to
find a vector $x^*$
minimizing an objective function $f$
over a search space $\mathcal{X}\subset\mathbb{R}^d$
while satisfying inequality constraints $g(x)\leq 0$
and equality constraints $h(x)=0$:

$$
\begin{align}
&\underset{x\in\mathcal{X}}{\operatorname{minimize}}& & f(x) \\
&\operatorname{subject\;to}
& &g(x) \leq 0 \\
&&&h(x) = 0
\end{align}
$$

where $f:\mathcal{X}\mapsto\mathbb{R}$,
$g:\mathcal{X}\mapsto\mathbb{R}^{p_g}$
and $h:\mathcal{X}\mapsto\mathbb{R}^{p_h}$.

Any optimization problem of the form

$$
\begin{align}
&\underset{x\in\mathcal{X}}{\operatorname{minimize}}& & f_{\textrm{cost}}(x) \\
&\underset{x\in\mathcal{X}}{\operatorname{maximize}}&
& f_{\textrm{performance}}(x) \\
&\operatorname{subject\;to} & &g_n(x) \leq t_{g_n} \\
&&&g_p(x) \geq t_{g_p} \\
&&&\tilde{h}(x) = t_h
\end{align}
$$

can be reduced to this standard optimization problem:

- an objective to minimize,
- upper inequality constraints with bounds equal to 0,
- equality constraints with right-hand sides equal to 0.

!!! note "API"

    In [GEMSEO](https://www.gemseo.org),
    the user instantiates an
    [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
    from a [DesignSpace][gemseo.algos.design_space.DesignSpace],
    defines its objective functions and constraints
    with
    [MDOFunction][gemseo.core.mdo_functions.mdo_function.MDOFunction] objects
    and solves it with an algorithm
    from a [BaseDriverLibrary][gemseo.algos.base_driver_library.BaseDriverLibrary].
    This algorithm can be either an optimizer or a design of experiments (DOE).

    ??? example

        The optimization problem

        $$
        \begin{align}
        &\underset{x\in[-1,1]}{\operatorname{minimize}}& & x^2 \\
        &\operatorname{subject\;to} & & x^3 \ge 0.1
        \end{align}
        $$

        can be solved with the Python lines

        ``` py
        from gemseo import execute_algo
        from gemseo.algos.optimization_problem import Optimization
        from gemseo.algos.design_space import DesignSpace
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        design_space = DesignSpace()
        design_space.add_variable("x", lower_bound=-1., upper_bound=1.)

        problem = OptimizationProblem(design_space)
        problem.objective = MDOFunction(lambda x: x**2, "f")
        problem.add_constraint(MDOFunction(lambda x: x**3, "g"), positive=True, value=0.1)

        execute_algo(problem, algo_name="PYDOE_FULLFACT", n_samples=10, algo_type="doe")
        ```

### Multidisciplinary optimization (MDO) problem

In complex systems,
the quantities $f(x)$, $g(x)$ and $h(x)$ are often computed by $D$ models,
called *disciplines*,
which can be weakly or strongly coupled.
The vector $x$ can be split into

- a sub-vector $x_0$ shared by at least two disciplines,
- sub-vectors $x_i$, each specific to a discipline $i$.

The problem is then called a multidisciplinary optimization (MDO) problem

$$
\begin{align}
&\underset{x\in\mathcal{X},y\in\mathcal{Y}}{\operatorname{minimize}}&&f(x,y)\\
&\operatorname{subject\;to}
& &g(x,y) \leq 0 \\
&&&h(x,y) = 0 \\
&&&y=c(x,y)
\end{align}
$$

where $c_i:x_0,x_i,y_{-i}\mapsto y_i$ represents the $i$-th discipline
with $y_{-i}=\{y_j, 1\leq j\neq i \leq D\}$.

This MDO problem implies that
the optimum $(x^*,y^*)$ must be multidisciplinary feasible,
i.e. satisfying the coupling equations $y^*=c(x^*,y^*)$.
Solving these equations is called a
*multidisciplinary analysis*
(MDA).

??? example "Example: MDA with linear disciplines"

    Let us consider a simple MDO problem with two linear disciplines given by
    $c_1: x_0,x_1,y_2 \mapsto x_0 + x_1 + y_2$
    and
    $c_2: x_0,x_2,y_1 \mapsto x_0 + x_2 + 2y_1$.
    Let us define the objective function as
    $f(x_0,x_1,x_2,y_1,y_2)=c_1(x_0,x_1,y_2)^2+c_2(x_0,x_2,y_1)^2$.
    The coupling equations are
    $c_1(x_0,x_1,y_2)=y_1$ and $c_2(x_0,x_2,y_1)=y_2$.
    In this linear case,
    they can be solved analytically:
    $y_1(x) = -2x_0-x_1-x_2$ and $y_2(x) = -3x_0-2x_1-x_2$.
    Then,
    the objective function output can be rewritten as a function of $x$ only:
    $f(x,y(x))=y_1(x)^2+y_2(x)^2$,
    and the MDO problem becomes a simple optimization problem.

In the case of non-linear disciplines,
the MDA can be solved with Newton's method or a fixed-point technique.

Last but not least,
the efficient resolution of an MDO problem involves
finding a suitable rewriting of the problem,
called *MDO formulation* or *architecture*[@MartinsSurvey].

- The MDF (multidisciplinary feasible) formulation is certainly the best-known.
  This architecture performs an MDA at each iteration of the optimization loop
  and is thus qualified as *coupled*.

    $$
    \begin{align}
    &\underset{x\in\mathcal{X}}{\operatorname{minimize}}&&f(x,y(x))\\
    &\operatorname{subject\;to}
    & &g(x,y(x)) \leq 0 \\
    &&&h(x,y(x)) = 0 \\
    &&&y(x)=c(x,y(x)).
    \end{align}
    $$

- The IDF (individual disciplinary feasible) formulation is also popular.
  This architecture evaluates the disciplines independently
  at each iteration of the optimization loop
  and is thus qualified as *uncoupled*.
  The multidisciplinary feasibility is ensured
  at convergence of the optimization algorithm
  by means of consistency constraints.

    $$
    \begin{align}
    &\underset{x\in\mathcal{X},\tilde{y}\in\mathcal{Y}}{\operatorname{minimize}}&&f(x,\tilde{y})\\
    &\operatorname{subject\;to}
    & &g(x,\tilde{y}) \leq 0 \\
    &&&h(x,\tilde{y}) = 0 \\
    &&&\tilde{y} = c(x,\tilde{y}).
    \end{align}
    $$

- Bi-level formulations[@Gazaix2019] split the optimization problem
  into a top-level optimization problem controlling the shared variable $x_0$
  and a collection of sub-optimization problems
  whose $i$-th controls the local variable $x_i$.

!!! note "API"

    [GEMSEO](https://www.gemseo.org) offers implementations for the
    [MDF][gemseo.formulations.mdf.MDF],
    [IDF][gemseo.formulations.idf.IDF]
    and [BiLevel][gemseo.formulations.bilevel.BiLevel] formulations.

    ??? example "Example: MDF applied to the Sellar problem"

        ``` py
          from gemseo import create_scenario
          from gemseo.algos.design_space import DesignSpace
          from gemseo.disciplines.analytic import AnalyticDiscipline

          disciplines = [
              AnalyticDiscipline({"y_1": "(z1**2+z2+x-0.2*y2)**0.5"}, "Sellar1"),
              AnalyticDiscipline({"y_2": "abs(y1)+z1+z2"}, "Sellar2"),
              AnalyticDiscipline(
                  {
                      "f": "x**2+z2+y**2+exp(-y_2)",
                      "c1": "3.16-y1**2",
                      "c2": "y2-24"
                  },
                  "SellarSystem"
              )
          ]

          design_space = DesignSpace()
          design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1)
          design_space.add_variable("z1", lower_bound=-10, upper_bound=10.0, value=4.0)
          design_space.add_variable("z2", lower_bound=0.0, upper_bound=10.0, value=3.0)

          scenario = create_scenario(disciplines, "MDF", "f", design_space)
          scenario.add_constraint("c1", "ineq")
          scenario.add_constraint("c2", "ineq")
          scenario.execute(algo_name="SLSQP", max_iter=100)
        ```

### Optimization problem under uncertainty

The models are often subject to uncertainties
There are different ways of classifying uncertainties
and different ways of modeling them.
However,
GEMSEO-UMDO is limited to the probability theory
for the sake of simplicity
and because this framework is the most popular and has proved its worth.
And so,
the uncertainties are modelled by random variables.

Then,
the objective $f(x,U)$ and the constraints $g(x,U)$ and $h(x,U)$,
where $U$ denotes random inputs,
are in turn random variables
and the standard optimization problem is replaced by

$$
\begin{align}
&\underset{x\in\mathcal{X}}{\operatorname{minimize}}& & \mathbb{K}_f[f(x,U)] \\
&\operatorname{subject\;to}
& &\mathbb{K}_g[g(x,U)] \leq 0 \\
&&&\mathbb{K}_h[h(x,U)] = 0
\end{align}
$$

where $\mathbb{K}_f$, $\mathbb{K}_g$ and $\mathbb{K}_h$ are statistics.

The statistic of a function $\phi$ can be
the expectation $\mathbb{E}[\phi(x,U)]$,
the standard deviation $\mathbb{S}[\phi(x,U)]$,
the variance $\mathbb{V}[\phi(x,U)]$,
a margin $\mathbb{E}[\phi(x,U)]+\kappa\times\mathbb{S}[\phi(x,U)]$
or a probability $\mathbb{P}[m \leq \phi(x,U)\leq M]$.
For the inequality constraints,
$\mathbb{K}_g[g(x,U)]$ could be
$\mathbb{P}[g(x,U)\geq\epsilon]$
or $\mathbb{P}[g(x,U)\geq 0]-\varepsilon$.
For the equality constraints,
$\mathbb{K}_h[h(x,U)]$ could be
$\mathbb{P}[|h(x,U)|\geq\epsilon]$.

!!! note

    When $\phi(x,U)$ is normally distributed,
    the margin $\mathbb{E}[\phi(x,U)]+q_\alpha\times\mathbb{S}[\phi(x,U)]$
    corresponds to the $\alpha$-quantile of $\phi(x,U)$
    where $q_\alpha$ is the $\alpha$-quantile
    of the standard Gaussian distribution.
    For that reason,
    2 or 3 are common candidates for $\kappa$
    as in this case,
    the margins correspond to
    the 0.975- and 0.999- quantiles of $\phi(x,U)$ respectively.

Typically,
a margin is applied to the objective to ensure a robust optimum $x^*$:

- a small value of $f(x^*,u)$ by minimizing $\mathbb{E}[f(x,U)]$,
- whatever the realization $u$ of $U$ by minimizing $\mathbb{S}[f(x,U)]$.

## API

Here is an outline of the API. Go to the examples for more information.

### Disciplines

When defining disciplines,
do not forget to declare the uncertain variables as input variables
so that the
[UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
and
[UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
can change their values.

!!! example

    Let us implement an [Discipline][gemseo.core.discipline.discipline.Discipline]
    outputting $f(x,U)=(x_1+U)^2+(x_2+U)^2$:
    ``` py
    from numpy import array
    from gemseo.core.discipline.discipline import Discipline


    class MyDiscipline(Discipline):

        def __init__(self):
            super().__init__()
            self.input_grammar.update_from_names(["x1", "x2", "U"])
            self.default_input_data = {"x1": array([0.]), "x2": array([0.]), "U": array([0.5])}

        def _run(self, input_data):
            x1 = self.io.data["x1"]
            x2 = self.io.data["x2"]
            U = self.io.data["U"]
            y = (x1+U)**2 + (x2+U)**2
            self.io.update_output_data({"y": y})
    ```

    This discipline can be executed
    with different values of the uncertain variable $U$:
    ``` py
    discipline.execute()  # default value, i.e. U=0.5
    discipline.execute({"U": array([0.2])})  # custom value: U=0.2
    ```

### Uncertain space

The uncertain variables have to be defined
in a [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
with the method [add_random_variable][gemseo.algos.parameter_space.ParameterSpace.add_random_variable].

!!! example

    In the previous example,
    we could model the uncertain variable $U$ as a random variable
    distributed according to a triangular distribution
    between 0.2 and 0.7 with a mode of 0.4:

    ``` py
    from gemseo.algos.parameter_space import ParameterSpace

    uncertain_space = ParameterSpace()
    uncertan_space.add_random_variable(
        "U", "OTTriangularDistribution", minimum=0.2, maximum=0.7, mode=0.4
    )
    ```

### Scenario

Given these disciplines and uncertain space and also a design space of course,
the MDO problem can be set up.

In the case of MDO *without* uncertainty,
there are two scenarios to set up the MDO problem:

- [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario]
  to solve it with a DOE,
- [MDOScenario][gemseo.scenarios.mdo_scenario.MDOScenario]
  to solve it with an optimizer.

Both need knowledge of objective and constraint functions
in addition to the disciplines and design space
to solve the MDO problem.

In the case of MDO *with* uncertainty,
there are two similar scenarios to set up the MDO problem:

- [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
  to solve it with a DOE,
- [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  to solve it with an optimizer.

Both need knowledge of the statistics and their estimators
in addition to the disciplines, design space, objective and constraints
to solve the MDO problem under uncertainty.

!!! note "API"

    The API of [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
    is deliberately similar to
    the API of [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario].
    And the same for
    [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
    and
    [MDOScenario][gemseo.scenarios.mdo_scenario.MDOScenario].
    This choice was made not only to simplify the user's life,
    but also because an MDO problem under uncertainty
    is first and foremost an MDO problem.

!!! example

    Continuing the previous example,
    we seek to minimize $\mathbf{E}[(x_1+U)^2+(x_2+U)^2]$
    over the domain $[-1,1]^2$
    with the gradient-free optimization algorithm COBYLA
    and a Monte Carlo estimator of the expectation.

    ``` py
    from gemseo.algos.design_space import DesignSpace
    from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
    from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

    design_space = DesignSpace()
    design_space.add_variable("x1", lower_bound=-1., upper_bound=1.)
    design_space.add_variable("x2", lower_bound=-1, upper_bound=1.)

    scenario = UMDOScenario(
        [discipline],
        "DisciplinaryOpt",
        "y",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation_settings=Sampling_Settings(n_samples=100),
    )
    scenario.execute(algo_name="NLOPT_COBYLA", max_iter=50)
    ```

### U-MDO formulations

[UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
can estimate the statistics associated with $f(x,U)$, $g(x,U)$ and $h(x,U)$
by sampling these random variables:

$$(f(x,U^{(i)}),g(x,U^{(i)}),h(x,U^{(i)}))_{1\leq i \leq N}.$$

However,
as sampling can be expensive,
GEMSEO-UMDO offers other techniques to reduce the cost of statistics estimation,
such as
[control variates](control_variate.md),
[Taylor polynomials](taylor_polynomial.md)
and
[polynomial chaos expansions](pce.md).
The choice of an estimation technique is made
via the argument `statistic_estimation_settings`
of [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
(or [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]),
which is a Pydantic model of U-MDO settings,
_e.g._
[ControlVariate_Settings][gemseo_umdo.formulations.control_variate_settings.ControlVariate_Settings],
[TaylorPolynomial_Settings][gemseo_umdo.formulations.taylor_polynomial_settings.TaylorPolynomial_Settings]
or [PCE_Settings][gemseo_umdo.formulations.pce_settings.PCE_Settings].

As of now,
only the [``Sampling`` U-MDO formulation](sampling.md) provides analytical derivatives of the statistics
when the disciplines and the multidisciplinary process generated by the MDO formulation are differentiable.
Therefore,
the other U-MDO formulations require gradient approximation to use gradient-based optimizers,
what can be expensive according to the dimension of the design space.

The rest of the **MDO under uncertainty** section of the user guide presents the different U-MDO formulations.

??? info "Implementation"

    Given a
    [DesignSpace][gemseo.algos.design_space.DesignSpace]
    and a collection of
    [Disciplines][gemseo.core.discipline.discipline.Discipline],
    a [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario]
    generates and solves an
    [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
    that corresponds to a
    [BaseMDOFormulation][gemseo.formulations.base_mdo_formulation.BaseMDOFormulation].
    The resolution consists in sampling the objective and constraints
    over the [DesignSpace][gemseo.algos.design_space.DesignSpace],
    i.e. $(x^{(i)},f(x^{(i)},U),g(x^{(i)},U),h(x^{(i)},U))_{1\leq i \leq N}$,
    and returning either the $x^*$ minimizing $f$ while satisying $g$ and $h$
    or the $x^*$ that violates the least $g$ and $h$.

    GEMSEO-UMDO uses this sampling mechanism a first time
    with a [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
    instead of the [DesignSpace][gemseo.algos.design_space.DesignSpace]
    to estimate the statistics
    $\mathbb{K}_f[f(x,U)]$, $\mathbb{K}_g[g(x,U)]$ and $\mathbb{K}_h[h(x,U)]$
    based on the samples
    $(U^{(i)},f(x,U^{(i)}),g(x,U^{(i)}),h(x,U^{(i)}))_{1\leq i \leq M}$.
    In the case of Monte Carlo sampling,
    $M$ is the number of samples
    while in the case of Taylor expansion,
    $M\in\{1,1+q\}$ where $q$ is the dimension of the uncertain space
    depending on whether the derivatives are known
    or to be estimated by finite differences.
    These statistics estimators
    $\hat{\mathbb{K}}_f[f(x,U)]$,
    $\hat{\mathbb{K}}_g[g(x,U)]$ and
    $\hat{\mathbb{K}}_h[h(x,U)]$
    are then used to build a new
    [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
    over the [DesignSpace][gemseo.algos.design_space.DesignSpace]:

    $$
    \begin{align}
    &\underset{x\in\mathcal{X}}{\operatorname{minimize}}
    & & \hat{\mathbb{K}}_f[f(x,U)] \\
    &\operatorname{subject\;to}
    & &\hat{\mathbb{K}}_g[g(x,U)] \leq 0 \\
    &&&\hat{\mathbb{K}}_h[h(x,U)] = 0.
    \end{align}
    $$

    Thus implemented,
    GEMSEO-UMDO should be able
    to set up any MDO problem under uncertainty
    from any [BaseMDOFormulation][gemseo.formulations.base_mdo_formulation.BaseMDOFormulation]
    and any statistic estimation technique.
    This vision may be theoretical at the moment,
    but the ambition of GEMSEO-UMDO is to be
    an engine generating U-MDO formulations
    based on any MDO formulation and any statistic estimators.
