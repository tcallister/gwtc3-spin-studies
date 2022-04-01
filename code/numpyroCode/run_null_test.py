import numpyro
from numpyro.infer import NUTS,MCMC
from jax import random
import numpy as np
import arviz as az
import json
from demo_likelihoods import gaussianDemo 

# Run over several chains to check convergence
nChains = 3
numpyro.set_host_device_count(nChains)

# Set RNG seeds
np.random.seed(1)
rng_key = random.PRNGKey(2)

# We'll repeatedly draw mock catalogs from a hypothetical Gaussian chi-effective distributions
# Specify the underlying properties of this hypothetical distribution here
mu_true = 0.06
sig_true = 0.09

# We also assign each mock observation a random measurement uncertainty, drawn from
# a Gaussian with the following mean and standard deviation.
# These quantities have been tuned to match the distribution of widths on the likelihoods
# for true chi-effective measurements in GWTC-3.
mean_logsig_obs = -0.9
std_logsig_obs = 0.3

# Loop across different catalog instantiations
runData = {}
for run in range(100):

    rng_key,rng_key_ = random.split(rng_key)

    # Instantiate dictionaries to hold things
    catalog = {}

    # Within each catalog, draw 69 events (matching the number of BBHs used in our actual sample)
    for i in range(69):

        # Draw a random measurement uncertainty
        logsig_obs = np.random.normal(loc=mean_logsig_obs,scale=std_logsig_obs)
        sig_obs = 10.**logsig_obs

        # Draw random true chi-effective values and observed max-likelihood values
        x_true = np.random.normal(loc=mu_true,scale=sig_true)
        x_ml = np.random.normal(loc=x_true,scale=sig_obs)

        # Record observation
        catalog[i] = {'x_true':x_true,
                        'x_ml':x_ml,
                        'sig_obs':sig_obs
                        }

    # Run inference
    kernel = NUTS(gaussianDemo)
    mcmc = MCMC(kernel,num_warmup=300,num_samples=3000,num_chains=nChains)
    mcmc.run(rng_key_,catalog)
    mcmc.print_summary()

    # Extract posterior chains
    data = az.from_numpyro(mcmc)
    stacked_samples = data.posterior.stack(draws=("chain", "draw"))
    posteriors = {'mu':stacked_samples.mu_chi.values.tolist(),
                    'logsig':stacked_samples.logsig_chi.values.tolist(),
                    'zeta_spike':stacked_samples.zeta_spike.values.tolist()}

    runData[run] = {'catalog':catalog,'posteriors':posteriors}

with open('../../data/effective_spin_null_test.json','w') as jf:
    json.dump(runData,jf)
