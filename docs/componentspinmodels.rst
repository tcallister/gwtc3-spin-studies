Component spin models
=====================

In addition to looking in the effective spin domain, we explore features in the underlying binary black hole component spin magnitude and tilt angle distributions using several different models.

Beta Plus Mixture |betaplusmixture-image|
-------------------------

.. |betaplusmixture-image| image:: images/model-cartoons-beta.pdf
    :width: 250
    
Following the GWTC-3 `Rates and Populations paper <https://arxiv.org/abs/2111.03634>`__, our simplest approach is to model the spin magnitudes :math:`\chi_i` as a beta distribution and the cosines of the spin tilt angles :math:`\cos\theta_i` as a mixture between aligned and isotropic subpopulations:

.. math::

     p(\chi_i | \alpha, \beta) \propto \chi_i^{1-\alpha} \, (1-\chi_i)^{1-\beta}

and 

.. math::

     p(\cos\theta|f_\mathrm{iso},\sigma_t) = \frac{f_\mathrm{iso}}{2} + (1-f_\mathrm{iso}){\mathcal {N}}_{[-1,1]}(\cos\theta|1,\sigma_t)\,.

Here the isotropic subpopulation is uniform on :math:`\cos\theta_i \in [-1,1]` and the aligned spin population is a truncated half gaussian :math: `\mathcal{N}` peaking at :math:`\cos\theta_i = 1` with a width :math:`\sigma_t`.

This model can be rerun as follows: 

.. code-block:: bash

    $ conda activate gwtc3-spin-studes
    $ cd code/emceeCode/
    $ python run_beta_plus_mixture.py
    
This code uses the :code:`betaPlusMixture` function:

.. autofunction:: emceeCode.posteriors.betaPlusMixture
    
The output will be a .json file storing the resulting posterior samples:

.. code-block:: bash

    data/component_spin_betaPlusMixture.json
    
.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/gwtc3-spin-studies/blob/main/data/inspect_component_spin_beta_plus_mixture.ipynb>`__.
    

Beta Plus Truncated Mixture |betaplustruncmixture-image|
-------------------------

.. |betaplustruncmixture-image| image:: images/model-cartoons-betaTruncated.pdf
    :width: 250
    
    
    
Beta Spike Plus Mixture |betaspikeplusmixture-image|
-------------------------

.. |betaspikeplusmixture-image| image:: images/model-cartoons-betaSpike.pdf
    :width: 250
    
    
    
Beta Spike Plus Truncated Mixture |betaspikeplustruncmixture-image|
-------------------------

.. |betaspikeplustruncmixture-image| image:: images/model-cartoons-betaSpikeTrunc.pdf
    :width: 250