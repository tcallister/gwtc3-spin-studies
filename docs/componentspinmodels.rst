Component spin models
=====================

We investigate potential features in the underlying binary black hole component spin magnitude and tilt angle distributions using several different models.

Beta + Mixture |betaplusmixture-image|
-----------------------------------------

.. |betaplusmixture-image| image:: images/model-cartoons-beta.pdf
    :width: 250
    
Following the GWTC-3 `Rates and Populations paper <https://arxiv.org/abs/2111.03634>`__, our simplest approach is to model the spin magnitudes :math:`\chi_i` as a beta distribution and the cosines of the spin tilt angles :math:`\cos\theta_i` as a mixture between aligned and isotropic subpopulations:

.. math::

     p(\chi | \alpha, \beta) \propto \chi^{1-\alpha} \, (1-\chi)^{1-\beta}

and 

.. math::

     p(\cos\theta|f_\mathrm{iso},\sigma_t) = \frac{f_\mathrm{iso}}{2} + (1-f_\mathrm{iso}){\mathcal {N}}_{[-1,1]}(\cos\theta|1,\sigma_t)\,.

Here the isotropic subpopulation is uniform over :math:`\cos\theta_i \in [-1,1]` and the aligned spin population is a truncated half gaussian :math:`\mathcal{N}` peaking at :math:`\cos\theta_i = 1` with a width :math:`\sigma_t`.

This model can be rerun as follows: 

.. code-block:: bash

    $ conda activate gwtc3-spin-studies
    $ cd code/emceeCode/
    $ python run_beta_plus_mixture.py
    
The output will be a .json file storing the resulting posterior samples:

.. code-block:: bash

    data/component_spin_betaPlusMixture.json
    
.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/gwtc3-spin-studies/blob/main/data/inspect_component_spin_beta_plus_mixture.ipynb>`__.
    

Beta + Truncated Mixture |betaplustruncmixture-image|
--------------------------------------------------------

.. |betaplustruncmixture-image| image:: images/model-cartoons-betaTruncated.pdf
    :width: 250
    
Here, we keep the same :math:`\chi_i` distribution as in Beta Plus Mixture but add a lower truncation bound :math:`z_\mathrm{min}` to the :math:`\cos\theta_i` distribution: 

.. math::

     p(\cos\theta|f_\mathrm{iso},\sigma_t,z_\mathrm{min}) = \frac{f_\mathrm{iso}}{1-z_\mathrm{min}} + (1-f_\mathrm{iso}){\mathcal {N}}_{[z_\mathrm{min},1]}(\cos\theta|1,\sigma_t)\,.

This model can be rerun as follows: 

.. code-block:: bash

    $ conda activate gwtc3-spin-studies
    $ cd code/emceeCode/
    $ python run_beta_plus_truncated_mixture.py
    
The output will be a .json file storing the resulting posterior samples:

.. code-block:: bash

    data/component_spin_betaPlusTruncatedMixture.json
    
.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/gwtc3-spin-studies/blob/main/data/inspect_component_spin_beta_plus_truncated_mixture.ipynb>`__.
    
    
    
Beta Spike + Mixture |betaspikeplusmixture-image|
----------------------------------------------------

.. |betaspikeplusmixture-image| image:: images/model-cartoons-betaSpike.pdf
    :width: 250
    
    
Here, we keep the same :math:`\cos\theta_i` distribution as in Beta Plus Mixture but add a spike to the :math:`\chi_i` distribution. This spike is a truncated half-gaussian centered at :math:`\chi=0`:

.. math::

    p(\chi | \alpha, \beta, f_\mathrm{spike}, \epsilon_\mathrm{spike}) = f_\mathrm{spike}{\mathcal {N}}_{[0,1]}(\chi|0,\epsilon_\mathrm{spike}) + \frac{1-f_\mathrm{spike}}{c(\alpha,\beta)} \chi_i^{1-\alpha} \, (1-\chi_i)^{1-\beta}

where :math:`f_\mathrm{spike}` is the fraction of the population in the spike, :math:`\epsilon_\mathrm{spike}` is the width of the spike, and :math:`c(\alpha,\beta)` is a normalization factor for the beta distribution (not a hyperparameter).

This model can be rerun as follows: 

.. code-block:: bash

    $ conda activate gwtc3-spin-studies
    $ cd code/emceeCode/
    $ python run_beta_spike_plus_mixture.py
    
The output will be a .json file storing the resulting posterior samples:

.. code-block:: bash

    data/component_spin_betaSpikePlusMixture.json
    
.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/gwtc3-spin-studies/blob/main/data/inspect_component_spin_beta_spike_plus_mixture.ipynb>`__.
    
    
    
Beta Spike + Truncated Mixture |betaspikeplustruncmixture-image|
-------------------------------------------------------------------

.. |betaspikeplustruncmixture-image| image:: images/model-cartoons-betaSpikeTrunc.pdf
    :width: 250
    
    
In our model with the most features, we use :math:`\chi_i` distribution from Beta Spike Plus Mixture and the :math:`\cos\theta_i` distribution from Beta Plus Truncated Mixture. 
 
This model can be rerun as follows: 

.. code-block:: bash

    $ conda activate gwtc3-spin-studies
    $ cd code/emceeCode/
    $ python run_beta_spike_plus_truncated_mixture.py
    
The output will be a .json file storing the resulting posterior samples:

.. code-block:: bash

    data/component_spin_betaSpikePlusTruncatedMixture.json
    
.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/gwtc3-spin-studies/blob/main/data/inspect_component_spin_beta_spike_plus_truncated_mixture.ipynb>`__.
    
    
.. autofunction:: emceeCode.posteriors.betaPlusMixture
.. autofunction:: emceeCode.posteriors.betaPlusTruncatedMixture
.. autofunction:: emceeCode.posteriors.betaSpikePlusMixture
.. autofunction:: emceeCode.posteriors.betaSpikePlusTruncatedMixture
