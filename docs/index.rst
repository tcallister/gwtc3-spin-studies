.. gwtc3-spin-studies documentation master file, created by
   sphinx-quickstart on Wed Mar 23 12:47:39 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gwtc3-spin-studies's documentation!
==============================================

This page details the code used to produce the results presented in *No evidence that the majority of black holes in binaries have zero spin*, which can be accessed at

https://github.com/tcallister/gwtc3-spin-studies/

In this paper, we systematically explored the effective and component spin distributions of binary black holes among the LIGO/Virgo GWTC-3 catalog.
In particular, we tried to answer the following core questions, which have been the subject of active exploration and some debate in the literature:

*Is there an excess of binary black holes with vanishing spin, as predicted by some theories of angular momentum transport in stellar cores?*

**We find no evidence for an excess of vanishing spin systems.**
This finding is confirmed by three complementary analyses: one relying only on the Bayes factors between spinning and non-spinning priors for each BBH observation,
one that seeks to model the distribution of effective aligned spins,
and one modeling the distribution of component spin magnitudes and misalignment angles.
Instead, we find BBH spin magnitudes to be consistent with a single, continuous distribution that remains finite at :math:`\chi=0`.

*Do there exist binaries with component spins misaligned by more than 90 degrees relative to their orbits?*

**We find a strong preference for the existence of such strongly misaligned spins.**
Our analysis of the BBH component spin distribution indicates that at least some component spins are misaligned from their orbits by more than 90 degrees.
This result is robust under a variety of modeling choices regarding both the distribution of component spin magnitudes and tilts.

Contents:

.. toctree::
    :maxdepth: 1

    getting-started
    effective-spin-models
    componentspinmodels
    making-figures
