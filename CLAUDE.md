<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The team-wide conventions (Python tooling, code style, docstrings, changelog policy,
the gemseo-provided pytest fixtures and the `assert_exception` helper) come from the
team-common `common.md`, which is loaded automatically at session start. This file only
records what is specific to this repository; do not duplicate `common.md` here.

## Project Overview

gemseo-umdo is a GEMSEO plugin for multidisciplinary design optimization under
uncertainty (U-MDO): it turns a problem whose objective and constraints are statistics
(mean, variance, margin, probability, ...) of outputs depending on random variables
into a deterministic optimization problem, by combining an MDO formulation with a
statistic estimation technique. It also provides advanced UQ (multilevel Monte Carlo,
control variates) and visualizations of uncertainty propagation.

Python `>=3.11, <3.15`. It depends on gemseo `develop` (GEMSEO 7), pulled from git
through `[tool.uv.sources]`, so the gemseo 7 paths of `common.md` apply as is.
GEMSEO discovers the plugin through the `gemseo_plugins` entry point of
`pyproject.toml`.

The project is generated from the `copier-gemseo` template (`.copier-answers.yml`);
the files it manages (`.justfile`, `.pre-commit-config.yaml`, `.ruff.toml`,
`.pytest.ini`, `.gitlab-ci.yml`, `pyproject.toml`, `mkdocs.yml`, ...) are updated
through `copier update`, not by hand.

## Commands

Recipes live in `.justfile` (run `just --list`). They all go through `uv`; the
interpreter is the one of `.python-version` (3.11), overridable with `UV_PYTHON`.

| Recipe | What it does |
|---|---|
| `just test [args]` | `uv run pytest` |
| `just coverage [args]` | Tests plus XML/HTML coverage reports |
| `just coverage-diff upstream/develop` | Coverage then `diff-cover` on the changed lines, `--fail-under=100` |
| `just check` | Installs the git hooks then runs `prek` on all files (ruff, rumdl, license headers, commitizen, uv-lock) |
| `just doc` | Serve the docs with mkdocs (sets `DOCSTRING_INHERITANCE_ENABLE`) |
| `just install` / `just fresh` | `uv sync`; `fresh` wipes `.venv` and the caches first |
| `just update` | `uv lock --upgrade` and update the git hooks |
| `just dist` / `just publish` | Build and check the wheel, then upload |

Run a single test or a selection with the usual pytest arguments:
`just test tests/formulations/test_sampling.py::test_fn`, `just test tests/ -k keyword`.
`pytest-xdist` is installed (`-n auto`).

## Testing

`tests/` mirrors `src/gemseo_umdo/`. `tests/conftest.py` only re-exports the shared
fixtures of `gemseo.util.testing.pytest_conftest`. `tests/formulations/conftest.py`
provides a toy problem (`disciplines`, `design_space`, `uncertain_space`,
`quadratic_problem`) and `statistic_estimation_settings`, parametrized over one
settings instance of every U-MDO formulation; `statistic_estimation_settings_for_dirac`
restricts it to the formulations compatible with a Dirac distribution. A new
formulation should be added to that parametrization.

Image tests use `syrupy-matplotlib`.

## Architecture

There are no lazy re-exports in the `__init__.py` files: import from the defining
module, e.g. `from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario`,
`from gemseo_umdo.formulations.sampling_settings import Sampling_Settings`.

### Scenarios

`scenarios/`: `UMDOScenario` (over `MDOScenario`, optimizer or DOE) and
`UEvaluationScenario` (over `EvaluationScenario`, DOE, observables only) mix
`BaseUScenario` into a GEMSEO scenario class. They take the disciplines, a `DesignSpace`, a `RandomSpace` (the
uncertain space) and the statistic estimation settings (a `*_Settings` of a U-MDO
formulation, passed to GEMSEO as the formulation settings); the inner MDO formulation
is chosen with `formulation_settings` (default `MDF_Settings`).
`add_observable` (both) and `add_objective`, `add_constraint` (`UMDOScenario` only)
take a `statistic_name` plus
`**statistic_parameters` (e.g. `factor` for `"Margin"`).

GEMSEO scenarios create their formulation through the class attribute
`_formulation_factory`; `BaseUScenario.__init__` temporarily shadows it with a
`SimpleNamespace` whose `create` binds `uncertain_space` and
`mdo_formulation_settings`, which the GEMSEO scenario does not pass.

`uncertain_design_variables` prepends a chain of noising disciplines
(`disciplines/`: `AdditiveNoiser`, `MultiplicativeNoiser`, found by class name or
`SHORT_NAME` through `NoiserFactory`) that computes `x` from a generated design
variable `dv_x` and an uncertain variable.

### U-MDO formulations

`formulations/`: every formulation derives from `BaseUMDOFormulation` and has a
sibling `<name>_settings.py` defining `<Name>_Settings` (deriving from
`BaseUMDOFormulationSettings`), found by `UMDO_FORMULATION_FACTORY`. The concrete ones
are `Sampling` (and `SequentialSampling`), `TaylorPolynomial`, `ControlVariate`,
`Surrogate` (and `PCE`).

`BaseUMDOFormulation` wraps an ordinary MDO formulation (`_mdo_formulation`) built over
an `EvaluationProblem` on the **uncertain** space, differentiable with respect to the
design variables. Formulations with `_USE_AUXILIARY_MDO_FORMULATION = True`
(`TaylorPolynomial`, `ControlVariate`, `PCE`) also build `_auxiliary_mdo_formulation`,
differentiable with respect to the uncertain variables. The MDO formulation is also
instantiated once over the design space only to update it (e.g. IDF adds the coupling
variables).

Each objective/constraint/observable becomes a `BaseStatisticFunction` (an
`ArrayFunction`, in `formulations/_functions/`) wrapping a statistic estimator.
Estimators live in `formulations/_statistics/<technique>/`, one module per statistic
(`mean.py`, `variance.py`, `standard_deviation.py`, `margin.py`, `probability.py`),
discovered by the `BaseFactory` of that package: the `statistic_name` given by the user
is the estimator **class name**, so a technique supports exactly the statistics its
package defines (PCE and Taylor polynomial have no `Probability`). A formulation
declares its pair either statically (`_STATISTIC_FACTORY`, `_STATISTIC_FUNCTION_CLASS`)
or at instantiation (`_statistic_factory`, `_statistic_function_class`), as `Sampling`
does to switch between one-shot and iterative estimation
(`estimate_statistics_iteratively`).

Samples or statistics computed at a design point are cached in
`input_data_to_output_data`, which a listener on the optimization problem clears when
the design point changes.

`UMDOFormulationsFactory` and `UMDO_FORMULATION_FACTORY` predate the naming rules of
`common.md` (see the `TODO: API` comment); renaming them is a breaking change.

### Other packages

| Package | Purpose |
|---|---|
| `statistics/multilevel/` | `MLMC` and `MLMCMLCV` estimators, standalone (not used by the formulations), with pilot-statistic factories |
| `monte_carlo_sampler.py` | `MonteCarloSampler` |
| `visualizations/` | `SobolGraph`, `UncertainCouplingGraph` |
| `problems/` | `UOptAsUMDOScenario`: a single-discipline problem rewritten as a U-MDO scenario |
| `use_cases/` | Beam, heat equation, spring-mass and truss models used by the docs and tests |
| `_utils/compatibility/` | Shims across numpy and openturns versions |

## Changelog and Docs

The changelog is the single `CHANGELOG.md` (Keep a Changelog sections, under a
`## Develop` heading until release); there are no fragments. Entries reference the API
with mkdocs cross-references and prefix breaking changes with `BREAKING CHANGE:`.

Docs are mkdocs: the user guide is in `docs/user_guide/`, examples are mkdocs-gallery
scripts under `docs/examples/{umdo,visualizations,problems}/`, the API pages are
generated by `docs/gen_ref_nav.py`, bibliography in `docs/references.bib`.
