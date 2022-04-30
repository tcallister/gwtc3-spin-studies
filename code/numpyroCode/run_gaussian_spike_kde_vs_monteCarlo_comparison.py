import numpyro
from numpyro.infer import NUTS,MCMC
from jax import random
import numpy as np
import arviz as az
import numpy as np
np.random.seed(151012)
from likelihoods import gaussianSpike,gaussianSpike_MonteCarloAvg
from getData import *

# Run over several chains to check convergence
nChains = 3
numpyro.set_host_device_count(nChains)

# Get dictionaries holding injections and posterior samples
mMin=5.
injectionDict = getInjections(mMin=mMin)
sampleDict = getSamples(mMin=mMin)

rng_key = random.PRNGKey(0)

# Loop across several spike widths
for sig_eps in [0.03,0.01,0.003,0.001,0.0003,0.0001]:

    rng_key,rng_key_ = random.split(rng_key)

    # For each spike width, run the Monte Carlo averaged likelihood
    kernel = NUTS(gaussianSpike_MonteCarloAvg)
    mcmc = MCMC(kernel,num_warmup=300,num_samples=3000,num_chains=nChains)
    mcmc.run(rng_key_,sampleDict,injectionDict,mMin,sig_eps)
    mcmc.print_summary()

    # Read out and save data
    data = az.from_numpyro(mcmc)
    az.to_netcdf(data,"../../data/kde_vs_monteCarlo_comparisons/effective_spin_gaussian_spike_mc_eps_{0:.4f}.cdf".format(sig_eps))

    rng_key,rng_key_ = random.split(rng_key)

    # Run the KDE'd likelihood
    kernel = NUTS(gaussianSpike)
    mcmc = MCMC(kernel,num_warmup=300,num_samples=3000,num_chains=nChains)
    mcmc.run(rng_key_,sampleDict,injectionDict,mMin,sig_eps)
    mcmc.print_summary()

    # Save data
    data = az.from_numpyro(mcmc)
    az.to_netcdf(data,"../../data/kde_vs_monteCarlo_comparisons/effective_spin_gaussian_spike_kde_eps_{0:.4f}.cdf".format(sig_eps))
