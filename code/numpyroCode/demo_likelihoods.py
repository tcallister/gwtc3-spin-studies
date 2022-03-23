import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.scipy.special import erf
from jax import vmap

def gaussianDemo(sampleDict,sig_eps):

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

    # Also sample the mixture fraction governing the number of events in the bulk Gaussian.
    # In order to faciliate more efficient sampling, we explicitly sample logit(zeta) rather than zeta directly.
    # This is then converted to zeta, and an appropriate term added to our log-likelihood to ensure
    # a uniform prior on zeta
    logit_zeta_bulk = numpyro.sample("logit_zeta_bulk",dist.Normal(0,2))
    zeta_bulk = jnp.exp(logit_zeta_bulk)/(1.+jnp.exp(logit_zeta_bulk)) 
    numpyro.deterministic("zeta_bulk",zeta_bulk)
    zeta_bulk_logprior = -0.5*logit_zeta_bulk**2/2**2 + jnp.log(1./zeta_bulk + 1./(1-zeta_bulk))
    numpyro.factor("uniform_zeta_bulk_prior",-zeta_bulk_logprior)

    # This function defines the per-event log-likelihood
    # x_sample: Mock likelihood samples 
    def logp(x_sample):
        
        # KDE likelihood; see paper text
        sig_kde = 0.5*jnp.std(x_sample)*x_sample.size**(-1./5.)
        bulk_denom = jnp.sqrt(2.*jnp.pi*(sig_kde**2+sig**2))*(erf((1.-mu)/jnp.sqrt(2.*sig**2)) + erf((1.+mu)/jnp.sqrt(2.*sig**2)))
        spike_denom = jnp.sqrt(2.*jnp.pi*(sig_kde**2+sig_eps**2))*(erf(1./jnp.sqrt(2.*sig_eps**2)) + erf(1./jnp.sqrt(2.*sig_eps**2)))
        bulk_kde_integrals = (erf((sig_kde**2*(1.+mu)+sig**2*(1.+x_sample))/jnp.sqrt(2.*sig_kde**2*sig**2*(sig_kde**2+sig**2)))\
                            - erf((sig_kde**2*(mu-1.)+sig**2*(x_sample-1.))/jnp.sqrt(2.*sig_kde**2*sig**2*(sig_kde**2+sig**2))))\
                        *jnp.exp(-(x_sample-mu)**2/(2.*(sig_kde**2+sig**2)))/bulk_denom
        spike_kde_integrals = (erf((sig_kde**2+sig_eps**2*(1.+x_sample))/jnp.sqrt(2.*sig_kde**2*sig_eps**2*(sig_kde**2+sig_eps**2)))\
                            - erf((sig_kde**2*(-1.)+sig_eps**2*(x_sample-1.))/jnp.sqrt(2.*sig_kde**2*sig_eps**2*(sig_kde**2+sig_eps**2))))\
                        *jnp.exp(-x_sample**2/(2.*(sig_kde**2+sig_eps**2)))/spike_denom
    
        # Form total population prior
        p_chi = zeta_bulk*bulk_kde_integrals + (1.-zeta_bulk)*spike_kde_integrals

        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(p_chi)**2/jnp.sum(p_chi**2)     
        return jnp.log(jnp.mean(p_chi)),n_eff
    
    # Map the log-likelihood function over each event in our catalog
    log_ps,n_effs = vmap(logp)(jnp.array([sampleDict[k] for k in sampleDict]))

    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))
