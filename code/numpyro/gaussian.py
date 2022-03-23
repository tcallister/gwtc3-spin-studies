import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.scipy.special import erf
from jax import vmap

def gaussian(sampleDict,injectionDict,mMin):

    """
    Implementation of a Gaussian effective spin distribution.

    INPUTs
    sampleDict: Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict: Precomputed dictionary containing successfully recovered injections
    mMin: Minimum black hole mass
    """
    
    # Sample our hyperparameters
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # mu: Mean of the chi-effective distribution
    # logsig_chi: Log10 of the chi-effective distribution's standard deviation
    bq = numpyro.sample("bq",dist.Normal(0,3))
    mu = numpyro.sample("mu_chi",dist.Uniform(-1,1))
    logsig_chi = numpyro.sample("logsig_chi",dist.Uniform(-1.5,0))
    sig = 10.**logsig_chi

    # Read out found injections
    # Note that `pop_reweight` is the inverse of the draw weights for each event
    Xeff_det = injectionDict['Xeff']
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    pop_reweight = injectionDict['pop_reweight']

    # Probability of each injection under the proposed population
    # See discussion of KDE likelihood methods in paper text
    sig_kde = 0.5*jnp.std(Xeff_det)*Xeff_det.size**(-1./5.)
    bulk_denom = jnp.sqrt(2.*jnp.pi*(sig_kde**2+sig**2))*(erf((1.-mu)/jnp.sqrt(2.*sig**2)) + erf((1.+mu)/jnp.sqrt(2.*sig**2)))
    bulk_kde_integral = (erf((sig_kde**2*(1.+mu)+sig**2*(1.+Xeff_det))/jnp.sqrt(2.*sig_kde**2*sig**2*(sig_kde**2+sig**2)))\
                        - erf((sig_kde**2*(mu-1.)+sig**2*(Xeff_det-1.))/jnp.sqrt(2.*sig_kde**2*sig**2*(sig_kde**2+sig**2))))\
                    *jnp.exp(-(Xeff_det-mu)**2/(2.*(sig_kde**2+sig**2)))/bulk_denom
    
    # Form ratio of proposed population weights over draw weights for each found injection
    p_chi_det = bulk_kde_integral
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq) - mMin**(1.+bq))
    p_m2_det = jnp.where(m2_det<mMin,0.,p_m2_det)
    xi_weights = p_chi_det*p_m2_det*pop_reweight
    
    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(xi_weights)**2/jnp.sum(xi_weights**2)
    nObs = 1.0*len(sampleDict)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)

    # Compute net detection efficiency and add to log-likelihood
    xi = jnp.sum(xi_weights)
    numpyro.factor("xi",-nObs*jnp.log(xi))
    
    # This function defines the per-event log-likelihood
    # m1_sample: Primary mass posterior samples
    # m2_sample: Secondary mass posterior samples
    # Xeff_sample: Effective spin posterior samples
    # weights: Factors that convert to the desired m1/redshift distribution and divide out the m2 and spin prior
    def logp(m1_sample,m2_sample,Xeff_sample,weights):
        
        # KDE likelihood; see paper text
        sig_kde = 0.5*jnp.std(Xeff_sample)*Xeff_sample.size**(-1./5.)
        bulk_denom = jnp.sqrt(2.*jnp.pi*(sig_kde**2+sig**2))*(erf((1.-mu)/jnp.sqrt(2.*sig**2)) + erf((1.+mu)/jnp.sqrt(2.*sig**2)))
        bulk_kde_integral = (erf((sig_kde**2*(1.+mu)+sig**2*(1.+Xeff_sample))/jnp.sqrt(2.*sig_kde**2*sig**2*(sig_kde**2+sig**2)))\
                            - erf((sig_kde**2*(mu-1.)+sig**2*(Xeff_sample-1.))/jnp.sqrt(2.*sig_kde**2*sig**2*(sig_kde**2+sig**2))))\
                        *jnp.exp(-(Xeff_sample-mu)**2/(2.*(sig_kde**2+sig**2)))/bulk_denom
    
        # Form total population prior
        p_chi = bulk_kde_integral
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq) - mMin**(1.+bq))
        p_m2 = jnp.where(m2_sample<mMin,0.,p_m2)
        mc_weights = p_chi*p_m2*weights
        
        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff
    
    # Map the log-likelihood function over each event in our catalog
    log_ps,n_effs = vmap(logp)(
                        jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['Xeff'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['weights_over_priors'] for k in sampleDict]))
        
    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))
