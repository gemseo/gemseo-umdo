..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Changelog titles are:
   - Added for new features.
   - Changed for changes in existing functionality.
   - Deprecated for soon-to-be removed features.
   - Removed for now removed features.
   - Fixed for any bug fixes.
   - Security in case of vulnerabilities.

Changelog
=========

All notable changes of this project will be documented here.

The format is based on
`Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_
and this project adheres to
`Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Version 1.0.1 (January 2023)
****************************

Changed
-------

- API change: the argument ``statistic_estimation_options`` of :class:`.UMDOFormulation` has been renamed to ``statistic_estimation_parameters``.
- API change: :attr:`.UMDOFormulation._processed_functions` replaces :attr:`.Sampling.processed_functions`.

Version 1.0.0 (July 2022)
*************************

First release.
