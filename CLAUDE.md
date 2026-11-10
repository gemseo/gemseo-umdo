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
| `just doc [--no-gallery] [-- args]` | Run `docs-tools prebuild` (`just doc-prebuild`) then serve the docs with Zensical (sets `DOCSTRING_INHERITANCE_ENABLE`); `--no-gallery` skips the examples |
| `just install` / `just fresh` | `uv sync`; `fresh` wipes `.venv` and the caches first |
| `just update` | `uv lock --upgrade` and update the git hooks |
| `just dist` / `just publish` | Build and check the wheel, then upload |

Run a single test or a selection with the usual pytest arguments:
`just test tests/formulation/test_sampling.py::test_fn`, `just test tests/ -k keyword`.
`pytest-xdist` is installed (`-n auto`).

## Testing

`tests/` mirrors `src/gemseo_umdo/`. `tests/conftest.py` only re-exports the shared
fixtures of `gemseo.util.testing.pytest_conftest`. `tests/formulation/conftest.py`
provides a toy problem (`disciplines`, `design_space`, `uncertain_space`,
`quadratic_problem`) and `statistic_estimation_settings`, parametrized over one
settings instance of every U-MDO formulation; `statistic_estimation_settings_for_dirac`
restricts it to the formulations compatible with a Dirac distribution. A new
formulation should be added to that parametrization.

Image tests use `syrupy-matplotlib`.

## Architecture

Three packages have eager public re-exports (with `__all__`), for the users:
`gemseo_umdo.formulation` (the settings of the U-MDO formulations; it imports only
`*_settings.py` modules, never a formulation class, to avoid import cycles),
`gemseo_umdo.statistic` (the concrete settings of the statistics, from
`statistic/settings.py`; `BaseStatisticSettings` is imported from that module) and
`gemseo_umdo.scenario` (`UMDOScenario`, `UEvaluationScenario`). The docs examples
import from them. Everywhere else, and in
`src/`, import from the defining module, e.g.
`from gemseo_umdo.formulation.sampling_settings import Sampling_Settings`.

### Scenarios

`scenario/`: `UMDOScenario` (over `MDOScenario`, optimizer or DOE) and
`UEvaluationScenario` (over `EvaluationScenario`, DOE, observables only) mix
`BaseUScenario` into a GEMSEO scenario class. They take the disciplines, a `DesignSpace`, a `RandomSpace` (the
uncertain space) and the statistic estimation settings (a `*_Settings` of a U-MDO
formulation, passed to GEMSEO as the formulation settings); the inner MDO formulation
is chosen with `formulation_settings` (default `MDF_Settings`).
`add_observable` (both) and `add_objective`, `add_constraint` (`UMDOScenario` only)
take a `statistic_settings`: a `*_Settings` of a statistic (e.g.
`Margin_Settings(factor=3.0)`) or a statistic name for its default settings (e.g.
`"Mean"`).

GEMSEO scenarios create their formulation through the class attribute
`_formulation_factory`; `BaseUScenario.__init__` temporarily shadows it with a
`SimpleNamespace` whose `create` binds `uncertain_space` and
`mdo_formulation_settings`, which the GEMSEO scenario does not pass.

`uncertain_design_variables` prepends a chain of noising disciplines
(`discipline/noiser/`: `AdditiveNoiser`, `MultiplicativeNoiser`, found by class name
or `short_name` through `NoiserFactory` in `discipline/noiser/factory.py`), built by
`create_noising_discipline_chain` of `discipline/noiser/util.py`, that computes `x`
from a generated design variable `dv_x` and an uncertain variable.

### U-MDO formulations

`formulation/`: every formulation derives from `BaseUMDOFormulation` (`formulation/core/base.py`) and has a
sibling `<name>_settings.py` defining `<Name>_Settings` (deriving from
`BaseUMDOFormulationSettings`), found by `umdo_formulation_factory`. The concrete ones
are `Sampling` (and `SequentialSampling`), `TaylorPolynomial`, `ControlVariate`,
`Surrogate` (and `PCE`).

`BaseUMDOFormulation` wraps an ordinary MDO formulation (`_mdo_formulation`) built over
an `EvaluationProblem` on the **uncertain** space, differentiable with respect to the
design variables. Formulations with `_use_auxiliary_mdo_formulation = True`
(`TaylorPolynomial`, `ControlVariate`, `PCE`) also build `_auxiliary_mdo_formulation`,
differentiable with respect to the uncertain variables. The MDO formulation is also
instantiated once over the design space only to update it (e.g. IDF adds the coupling
variables).

Each objective/constraint/observable becomes a `BaseStatisticFunction` (an
`ArrayFunction`, in `formulation/_function/`) wrapping a statistic estimator.
Estimators live in `formulation/_estimator/<technique>/` (base class in `base.py`), one module per statistic
(`mean.py`, `variance.py`, `standard_deviation.py`, `margin.py`, `probability.py`,
`quantile.py`), discovered by the `BaseFactory` of that package. A statistic is set by
a settings model of `statistic/settings.py` deriving from
`BaseStatisticSettings`; its `target_class_name` (`Margin` for `Margin_Settings`) is
the estimator **class name**, and the settings are passed to the estimator
constructor as `settings` (`BaseFactory.create_from_settings`, stored in `_settings`
and read from there, never copied into attributes; the estimators used internally,
e.g. the mean of a margin, get default settings). `BaseStatisticEstimator` is
`Generic[T]` over the settings type: the technique bases stay generic, the concrete
estimators bind it (`Margin(BaseSamplingEstimator[Margin_Settings])`), and `Variance`
stays generic where `StandardDeviation` derives from it
(`StandardDeviation(Variance[StandardDeviation_Settings])`). The
formulation copies the settings of the user. So a technique supports exactly the statistics its package
defines (PCE and Taylor polynomial have no `Probability`, only the one-shot `sampling/`
package has `Quantile`). The settings hold the defaults and the validation;
`make_conservative` makes them conservative for an objective or inequality constraint
(e.g. the `factor` of a margin, the `level` of a quantile); `get_name_parameters`
gives the fields used in the function name (those of `_name_field_names`, e.g. the
factor of a margin, plus those set explicitly). `BaseStatisticSettings.from_name`
resolves a string. An estimator exposes `min_n_samples`.

A formulation sets `_statistic_function_class` at class level and `_statistic_factory`
at class level or at instantiation before calling `super().__init__`. `Sampling` picks
the iterative or one-shot factory from `estimate_statistics_iteratively` and raises an
error for a statistic that cannot be estimated iteratively (`Quantile`) when this
setting is `True`. `_register_statistic_function`, called at the end of
`create_objective`, `create_constraint`, `add_constraint` and `add_observable`, makes
the function take part in the sampling.

Samples or statistics computed at a design point are cached in
`input_data_to_output_data`, which a listener on the optimization problem clears when
the design point changes.

### Other packages

| Package | Purpose |
|---|---|
| `statistic/multilevel/` | `MLMC` and `MLMCMLCV` estimators, standalone (not used by the formulations), with pilot-statistic factories |
| `monte_carlo_sampler.py` | `MonteCarloSampler` |
| `visualization/` | `SobolGraph`, `UncertainCouplingGraph` |
| `problem/` | `UOptAsUMDOScenario` (a single-discipline problem rewritten as a U-MDO scenario), and the beam, heat equation, spring-mass and truss models used by the docs and tests |
| `_util/compatibility/` | Shims across numpy and openturns versions |

## Changelog and Docs

The changelog is the single `CHANGELOG.md` (Keep a Changelog sections, under a
`## Develop` heading until release); there are no fragments. Entries reference the API
with mkdocs cross-references and prefix breaking changes with `BREAKING CHANGE:`.

Docs are built by Zensical, configured by `mkdocs.yml`: the user guide is in
`docs/user_guide/`, examples are mkdocs-gallery scripts under
`docs/examples/{umdo,visualizations,problems}/`, the API pages are generated by the
`api-autonav` plugin, bibliography in `docs/references.bib`. Zensical supports neither
MkDocs hooks nor the mkdocs-gallery and gen-files plugins, so the command
`docs-tools prebuild` runs the gallery and writes `_docs/generated.yml` (the
navigation of `nav.yml` with the galleries expanded, inherited by `mkdocs.yml`) and
`_docs/credits.md` (included by `docs/credits.md`) before every build; edit the navigation in `nav.yml`. `zensical serve` does not run
the prebuild, so run `just doc-prebuild` again after a change of `nav.yml` or of an
example. An example runs again when its file changes, and all of them when a file of `src` or `uv.lock` changes
(an MD5 stored in `docs/generated/examples/library.md5`); `just doc --no-gallery`
skips the examples, and the Zensical options go after `--`. The
`[@key]` citations are rendered by the Python-Markdown extension of
the `docs-tools` package, which also provides the prebuild command, from the repository
[gemseo/dev/doc/docs-tools](https://gitlab.com/gemseo/dev/doc/docs-tools).
