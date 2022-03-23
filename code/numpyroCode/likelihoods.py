import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.scipy.special import erf
from jax import vmap

def asym(x):
    return -jnp.exp(-x**2)/jnp.sqrt(np.pi)/x*(1.-1./(2.*x**2))

def truncatedNormal(samples,mu,sigma,lowCutoff,highCutoff):
    a = (lowCutoff-mu)/jnp.sqrt(2*sigma**2)
    b = (highCutoff-mu)/jnp.sqrt(2*sigma**2)
    norm = jnp.sqrt(sigma**2*np.pi/2)*(-erf(a) + erf(b))
    return jnp.exp(-(samples-mu)**2/(2.*sigma**2))/norm

def gaussian(sampleDict,injectionDict,mMin):

    """
    Implementation of a Gaussian effective spin distribution for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    mMin : float
        Minimum black hole mass
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

def doubleGaussian(sampleDict,injectionDict,mMin):

    """
    Implementation of an effective spin distribution described as a mixture of two Gaussians,
    for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    mMin : float
        Minimum black hole mass
    """
    
    # Sample our hyperparameters
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # mu_high: Mean of the dominant component of the chi-effective distribution
    # logsig_chi_high: Log10 of the dominant component's standard deviation
    # mu_low: Mean of the subdominant component
    # logsig_chi_log: Log10 of the subdominant component's standard deviation
    # zeta_high: Mixture fraction of events comprising the dominant population

    bq = numpyro.sample("bq",dist.Normal(0,3))
    mu_high = numpyro.sample("mu_chi_high",dist.Uniform(-1.,1))
    logsig_chi_high = numpyro.sample("logsig_chi_high",dist.Uniform(-1.5,0))
    mu_low = numpyro.sample("mu_chi_low",dist.Uniform(-1.,1.))
    logsig_chi_low = numpyro.sample("logsig_chi_low",dist.Uniform(-1.5,0))
    zeta_high = numpyro.sample("zeta_high",dist.Uniform(0.5,1))
    sig_high = 10.**logsig_chi_high
    sig_low = 10.**logsig_chi_low

    # Read out found injections
    # Note that `pop_reweight` is the inverse of the draw weights for each event
    Xeff_det = injectionDict['Xeff']
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    pop_reweight = injectionDict['pop_reweight']

    # Probability of each injection under the proposed population
    # See discussion of KDE likelihood methods in paper text
    sig_kde = 0.5*jnp.std(Xeff_det)*Xeff_det.size**(-1./5.)
    high_denom = jnp.sqrt(2.*jnp.pi*(sig_kde**2+sig_high**2))*(erf((1.-mu_high)/jnp.sqrt(2.*sig_high**2)) + erf((1.+mu_high)/jnp.sqrt(2.*sig_high**2)))
    low_denom = jnp.sqrt(2.*jnp.pi*(sig_kde**2+sig_low**2))*(erf((1.-mu_low)/jnp.sqrt(2.*sig_low**2)) + erf((1.+mu_low)/jnp.sqrt(2.*sig_low**2)))
    high_kde_integrals = (erf((sig_kde**2*(1.+mu_high)+sig_high**2*(1.+Xeff_det))/jnp.sqrt(2.*sig_kde**2*sig_high**2*(sig_kde**2+sig_high**2)))\
                        - erf((sig_kde**2*(mu_high-1.)+sig_high**2*(Xeff_det-1.))/jnp.sqrt(2.*sig_kde**2*sig_high**2*(sig_kde**2+sig_high**2))))\
                    *jnp.exp(-(Xeff_det-mu_high)**2/(2.*(sig_kde**2+sig_high**2)))/high_denom
    low_kde_integrals = (erf((sig_kde**2*(1.+mu_low)+sig_low**2*(1.+Xeff_det))/jnp.sqrt(2.*sig_kde**2*sig_low**2*(sig_kde**2+sig_low**2)))\
                        - erf((sig_kde**2*(mu_low-1.)+sig_low**2*(Xeff_det-1.))/jnp.sqrt(2.*sig_kde**2*sig_low**2*(sig_kde**2+sig_low**2))))\
                    *jnp.exp(-(Xeff_det-mu_low)**2/(2.*(sig_kde**2+sig_low**2)))/low_denom
    
    # Form ratio of proposed population weights over draw weights for each found injection
    p_chi_det = zeta_high*high_kde_integrals + (1.-zeta_high)*low_kde_integrals
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq) - mMin**(1.+bq))
    p_m2_det = jnp.where(m2_det<mMin,0.,p_m2_det)
    xi_weights = p_chi_det*p_m2_det*pop_reweight
    
    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(xi_weights)**2/jnp.sum(xi_weights**2)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/69.)

    # Compute net detection efficiency and add to log-likelihood
    xi = jnp.sum(xi_weights)
    numpyro.factor("xi",-69*jnp.log(xi))
    
    # This function defines the per-event log-likelihood
    # m1_sample: Primary mass posterior samples
    # m2_sample: Secondary mass posterior samples
    # Xeff_sample: Effective spin posterior samples
    # weights: Factors that convert to the desired m1/redshift distribution and divide out the m2 and spin prior
    def logp(m1_sample,m2_sample,Xeff_sample,weights):
        
        # KDE likelihood; see paper text
        sig_kde = 0.5*jnp.std(Xeff_sample)*Xeff_sample.size**(-1./5.)
        high_denom = jnp.sqrt(2.*jnp.pi*(sig_kde**2+sig_high**2))*(erf((1.-mu_high)/jnp.sqrt(2.*sig_high**2)) + erf((1.+mu_high)/jnp.sqrt(2.*sig_high**2)))
        low_denom = jnp.sqrt(2.*jnp.pi*(sig_kde**2+sig_low**2))*(erf((1.-mu_low)/jnp.sqrt(2.*sig_low**2)) + erf((1.+mu_low)/jnp.sqrt(2.*sig_low**2)))
        high_kde_integrals = (erf((sig_kde**2*(1.+mu_high)+sig_high**2*(1.+Xeff_sample))/jnp.sqrt(2.*sig_kde**2*sig_high**2*(sig_kde**2+sig_high**2)))\
                            - erf((sig_kde**2*(mu_high-1.)+sig_high**2*(Xeff_sample-1.))/jnp.sqrt(2.*sig_kde**2*sig_high**2*(sig_kde**2+sig_high**2))))\
                        *jnp.exp(-(Xeff_sample-mu_high)**2/(2.*(sig_kde**2+sig_high**2)))/high_denom
        low_kde_integrals = (erf((sig_kde**2*(1.+mu_low)+sig_low**2*(1.+Xeff_sample))/jnp.sqrt(2.*sig_kde**2*sig_low**2*(sig_kde**2+sig_low**2)))\
                            - erf((sig_kde**2*(mu_low-1.)+sig_low**2*(Xeff_sample-1.))/jnp.sqrt(2.*sig_kde**2*sig_low**2*(sig_kde**2+sig_low**2))))\
                        *jnp.exp(-(Xeff_sample-mu_low)**2/(2.*(sig_kde**2+sig_low**2)))/low_denom
        
        # Form total population prior
        p_chi = zeta_high*high_kde_integrals + (1.-zeta_high)*low_kde_integrals
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

def gaussianSpike_MonteCarloAvg(sampleDict,injectionDict,mMin,sig_eps):

    """
    Implementation of a Gaussian and zero-spin spike effective spin distribution for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    mMin : float
        Minimum black hole mass
    sig_eps : float
        Width of "spike" mixture component. We generally choose `sig_eps=0`
    """
    
    # Sample our hyperparameters
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # mu: Mean of the chi-effective distribution
    # logsig_chi: Log10 of the chi-effective distribution's standard deviation
    bq = numpyro.sample("bq",dist.Normal(0,3))
    mu = numpyro.sample("mu_chi",dist.Uniform(-1,1))
    numpyro.factor("mu_prior",-mu**2/(2.*0.4**2)) 
    logsig_chi = numpyro.sample("logsig_chi",dist.Uniform(-1.5,0.))
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
    
    # Read out found injections
    # Note that `pop_reweight` is the inverse of the draw weights for each event
    Xeff_det = injectionDict['Xeff']
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    pop_reweight = injectionDict['pop_reweight']

    # Form ratio of proposed population weights over draw weights for each found injection
    p_chi_det = zeta_bulk*truncatedNormal(Xeff_det,mu,sig,-1,1) + (1.-zeta_bulk)*truncatedNormal(Xeff_det,0.,sig_eps,-1,1)
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq) - mMin**(1.+bq))
    p_m2_det = jnp.where(m2_det<mMin,0.,p_m2_det)
    xi_weights = p_chi_det*p_m2_det*pop_reweight
    
    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(xi_weights)**2/jnp.sum(xi_weights**2)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/66.)

    # Compute net detection efficiency and add to log-likelihood
    xi = jnp.sum(xi_weights)
    numpyro.factor("xi",-66*jnp.log(xi))
    
    # This function defines the per-event log-likelihood
    # m1_sample: Primary mass posterior samples
    # m2_sample: Secondary mass posterior samples
    # Xeff_sample: Effective spin posterior samples
    # weights: Factors that convert to the desired m1/redshift distribution and divide out the m2 and spin prior
    def logp(m1_sample,m2_sample,Xeff_sample,weights):
        
        # Form total population prior
        p_chi = zeta_bulk*truncatedNormal(Xeff_sample,mu,sig,-1,1) + (1.-zeta_bulk)*truncatedNormal(Xeff_sample,0.,sig_eps,-1,1)
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
