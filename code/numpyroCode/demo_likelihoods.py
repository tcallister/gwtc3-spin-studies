import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.scipy.special import erf
from jax import vmap

def gaussianDemo(catalog):

    """
    Implementation of a Gaussian+Spike distribution for inference within `numpyro`.
    This model is used for demonstration purposes, yielding the "null" `zeta_spike` posteriors
    that we compare against our measured spike fraction with GWTC-3.

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing likelihood samples for each event in our mock catalog
    sig_eps : float
        Width of "spike" mixture component.
    """
    
    # Sample our hyperparameters
    # mu: Mean of the chi-effective distribution
    # logsig_chi: Log10 of the chi-effective distribution's standard deviation
    mu = numpyro.sample("mu_chi",dist.Uniform(-1,1))
    logsig_chi = numpyro.sample("logsig_chi",dist.Uniform(-1.5,0))
    sig = 10.**logsig_chi

    # Also sample the mixture fraction governing the number of events in the spike Gaussian.
    # In order to faciliate more efficient sampling, we explicitly sample logit(zeta) rather than zeta directly.
    # This is then converted to zeta, and an appropriate term added to our log-likelihood to ensure
    # a uniform prior on zeta
    logit_zeta_spike = numpyro.sample("logit_zeta_spike",dist.Normal(0,2))
    zeta_spike = jnp.exp(logit_zeta_spike)/(1.+jnp.exp(logit_zeta_spike)) 
    numpyro.deterministic("zeta_spike",zeta_spike)
    zeta_spike_logprior = -0.5*logit_zeta_spike**2/2**2 + jnp.log(1./zeta_spike + 1./(1-zeta_spike))
    numpyro.factor("uniform_zeta_spike_prior",-zeta_spike_logprior)

    # This function defines the per-event log-likelihood
    # x_sample: Mock likelihood samples 
    def logp(x_obs,sigma_obs):
        
        # KDE likelihood; see paper text
        sig_eps=0
        bulk_denom = jnp.sqrt(2.*jnp.pi*(sigma_obs**2+sig**2))*(erf((1.-mu)/jnp.sqrt(2.*sig**2)) + erf((1.+mu)/jnp.sqrt(2.*sig**2)))
        spike_denom = jnp.sqrt(2.*jnp.pi*(sigma_obs**2+sig_eps**2))*(erf(1./jnp.sqrt(2.*sig_eps**2)) + erf(1./jnp.sqrt(2.*sig_eps**2)))
        bulk_kde_integrals = (erf((sigma_obs**2*(1.+mu)+sig**2*(1.+x_obs))/jnp.sqrt(2.*sigma_obs**2*sig**2*(sigma_obs**2+sig**2)))\
                            - erf((sigma_obs**2*(mu-1.)+sig**2*(x_obs-1.))/jnp.sqrt(2.*sigma_obs**2*sig**2*(sigma_obs**2+sig**2))))\
                        *jnp.exp(-(x_obs-mu)**2/(2.*(sigma_obs**2+sig**2)))/bulk_denom
        spike_kde_integrals = (erf((sigma_obs**2+sig_eps**2*(1.+x_obs))/jnp.sqrt(2.*sigma_obs**2*sig_eps**2*(sigma_obs**2+sig_eps**2)))\
                            - erf((sigma_obs**2*(-1.)+sig_eps**2*(x_obs-1.))/jnp.sqrt(2.*sigma_obs**2*sig_eps**2*(sigma_obs**2+sig_eps**2))))\
                        *jnp.exp(-x_obs**2/(2.*(sigma_obs**2+sig_eps**2)))/spike_denom
    
        # Form total population prior
        p_chi = (1.-zeta_spike)*bulk_kde_integrals + zeta_spike*spike_kde_integrals

        # Return log-likelihood
        return jnp.log(p_chi)
    
    # Map the log-likelihood function over each event in our catalog
    log_ps = vmap(logp)(
            jnp.array([catalog[k]['x_ml'] for k in catalog]),
            jnp.array([catalog[k]['sig_obs'] for k in catalog]))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))
