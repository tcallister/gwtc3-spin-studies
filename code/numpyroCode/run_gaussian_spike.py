import numpyro
from numpyro.infer import NUTS,MCMC
from jax import random
import arviz as az
from likelihoods import gaussianSpike
from getData import *

# Run over several chains to check convergence
nChains = 3
numpyro.set_host_device_count(nChains)

# Get dictionaries holding injections and posterior samples
mMin=5.
injectionDict = getInjections(mMin=mMin)
sampleDict = getSamples(mMin=mMin)

# Set up NUTS sampler over our likelihood
kernel = NUTS(gaussianSpike)
mcmc = MCMC(kernel,num_warmup=300,num_samples=3000,num_chains=nChains)

# Choose a random key
rng_key = random.PRNGKey(2)
rng_key,rng_key_ = random.split(rng_key)

# Run our model.
# The final argument specifies that our "spike" has zero width, e.g. is a delta function
mcmc.run(rng_key_,sampleDict,injectionDict,mMin,0.)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"../../data/effective_spin_gaussian_spike.cdf")

