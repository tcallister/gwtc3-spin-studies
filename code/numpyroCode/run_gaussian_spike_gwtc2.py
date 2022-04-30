import numpyro
from numpyro.infer import NUTS,MCMC
from jax import random
import arviz as az
import numpy as np
np.random.seed(191109)
from likelihoods import gaussianSpike
from getData import *
import sys

# Run over several chains to check convergence
nChains = 3
numpyro.set_host_device_count(nChains)

# Get dictionaries holding injections and posterior samples
mMin=5.
injectionDict = getInjections(mMin=mMin)
sampleDict = getSamples(mMin=mMin)

events = list(sampleDict.keys())
for event in events:

    # Strip out O3b events
    if event[:4]=="S200" or event[:5]=="S1911" or event[:5]=="S1912":
        sampleDict.pop(event)

    # Also strip out GWTC-2.1 events
    elif event=="S190805bq" or event=="S190925ad":
        sampleDict.pop(event)

# Set up NUTS sampler over our likelihood
kernel = NUTS(gaussianSpike)
mcmc = MCMC(kernel,num_warmup=300,num_samples=3000,num_chains=nChains)

# Choose a random key
rng_key = random.PRNGKey(1)
rng_key,rng_key_ = random.split(rng_key)

# Run our model.
# The final argument specifies that our "spike" has zero width, e.g. is a delta function
mcmc.run(rng_key_,sampleDict,injectionDict,mMin,0.)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"../../data/effective_spin_gaussian_spike_gwtc2.cdf")

