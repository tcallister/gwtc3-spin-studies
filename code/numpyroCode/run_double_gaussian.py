import numpyro
from numpyro.infer import NUTS,MCMC
from jax import random
import numpy as np
import arviz as az
import sys
from likelihoods import doubleGaussian
from getData import *

# Run over several chains to check convergence
nChains = 3
numpyro.set_host_device_count(nChains)

# Get dictionaries holding injections and posterior samples
mMin = 5.
injectionDict = getInjections(mMin=mMin)
sampleDict = getSamples(mMin=mMin)

# Set up NUTS sampler over our likelihood
kernel = NUTS(doubleGaussian)
mcmc = MCMC(kernel,num_warmup=200,num_samples=2000,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(2)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict,mMin)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"../../data/effective_spin_doubleGaussians.cdf")

