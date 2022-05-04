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

    data/effective_spin_gaussian.cdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/gwtc3-spin-studies/blob/main/data/inspect_effective_spin_gaussian.ipynb>`__

GaussianSpike |gaussian-spike-image|
------------------------------------

.. |gaussian-spike-image| image:: images/model-cartoons-GaussianSpike.pdf
    :width: 100

To explore the possible presence of a zero-spin subpopulation, we extend the Gaussian model with a narrow (possibly delta-function) "spike" at :math:`\chi_\mathrm{eff}=0`:

.. math::

    p(\chi_\mathrm{eff}) = \zeta_\mathrm{spike} N(\chi_\mathrm{eff}|0,\epsilon_\mathrm{spike}) + (1 - \zeta_\mathrm{spike}) N(\chi_\mathrm{eff}|\mu_\mathrm{eff},\sigma_\mathrm{eff}) \qquad (-1\leq\chi_\mathrm{eff}\leq 1)

In most cases, we will take :math:`\epsilon_\mathrm{spike}=0` and let the "spike" population become a true delta function (see Appendix D of our paper text to learn about the KDE trick that allows us to evaluate the delta function's likelihood).
Occasionally, though, we will let :math:`\epsilon` be non-zero in order to check the appropriate convergence of our results; see Fig. 10.


This model can be rerun as follows:

.. code-block:: bash

    $ conda activate gwtc3-spin-studes
    $ cd code/numpyroCode/
    $ python run_gaussian_spike.py

The output will be a .cdf file storing the resulting posterior samples and diagnostic information:

.. code-block:: bash

    output/effective_spin_gaussian_spike.cdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/gwtc3-spin-studies/blob/main/data/inspect_effective_spin_gaussian.ipynb>`__

We include an analogous script, :code:`run_gaussian_spike_gwtc2.py`, to run the GaussianSpike model over only those events included in GWTC-2, in order to better compare with past results.

DoubleGaussian |double-gaussian-image|
--------------------------------------

.. |double-gaussian-image| image:: images/model-cartoons-twoGaussian.pdf
    :width: 100

.. math::

    p(\chi_\mathrm{eff}) = \zeta_a N(\chi_\mathrm{eff}|\mu_{\mathrm{eff},a}\sigma_{\mathrm{eff},a}) + (1-\zeta_a)N(\chi_\mathrm{eff}|\mu_{\mathrm{eff},b},\sigma_{\mathrm{eff},b}) \qquad (-1\leq\chi_\mathrm{eff}\leq 1)

This model can be rerun as follows:

.. code-block:: bash

    $ conda activate gwtc3-spin-studes
    $ cd code/numpyroCode/
    $ python run_double_gaussian.py

The output will be a .cdf file storing the resulting posterior samples and diagnostic information:

.. code-block:: bash

    data/effective_spin_doubleGaussians.cdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found here_.

.. _here: https://github.com/tcallister/gwtc3-spin-studies/blob/main/data/inspect_effective_spin_doubleGaussian.ipynb

.. autofunction:: numpyroCode.likelihoods.gaussian

.. autofunction:: numpyroCode.likelihoods.doubleGaussian

.. autofunction:: numpyroCode.likelihoods.gaussianSpike

.. autofunction:: numpyroCode.likelihoods.gaussianSpike_MonteCarloAvg
