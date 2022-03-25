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

This file can be read in and explored using the `arviz` module in python.
A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found HERE.


.. autofunction:: numpyroCode.likelihoods.gaussian

.. autofunction:: numpyroCode.likelihoods.doubleGaussian

.. autofunction:: numpyroCode.likelihoods.gaussianSpike

.. autofunction:: numpyroCode.likelihoods.gaussianSpike_MonteCarloAvg
