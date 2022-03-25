Effective spin models
=====================

We explore features in the binary black hole effective aligned spin distribution using several different models.

Gaussian |gaussian-image|
-------------------------

.. |gaussian-image| image:: images/model-cartoons-Gaussian.pdf
    :width: 100

Our simplest approach is just to model the BBH :math:`\chi_\mathrm{eff}` distribution as a Gaussian:

.. math::

    p(\chi_\mathrm{eff}) = N(\chi_\mathrm{eff}|\mu_\mathrm{eff},\sigma_\mathrm{eff}) \qquad (-1\leq\chi_\mathrm{eff}\leq 1)

This model can be rerun as follows:

.. code-block:: bash

    $ conda activate gwtc3-spin-studes
    $ cd code/numpyroCode/
    $ python run_gaussian.py

The output will be a .cdf file storing the resulting posterior samples and diagnostic information:

.. code-block:: bash

    output/effective_spin_gaussian.cdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found here_.

.. _here: https://github.com/tcallister/gwtc3-spin-studies/blob/main/data/inspect_effective_spin_gaussian.ipynb

GaussianSpike |gaussian-spike-image|
------------------------------------

.. |gaussian-spike-image| image:: images/model-cartoons-GaussianSpike.pdf
    :width: 100

...

DoubleGaussian |double-gaussian-image|
------------------------------------

.. |double-gaussian-image| image:: images/model-cartoons-twoGaussian.pdf
    :width: 100

...

.. autofunction:: numpyroCode.likelihoods.gaussian

.. autofunction:: numpyroCode.likelihoods.doubleGaussian

.. autofunction:: numpyroCode.likelihoods.gaussianSpike

.. autofunction:: numpyroCode.likelihoods.gaussianSpike_MonteCarloAvg
