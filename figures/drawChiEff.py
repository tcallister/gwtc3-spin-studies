import numpy as np
import sys
sys.path.append('./../code/emceeCode')
from posterior_helper_functions import *

def drawChieffs(mu_chi, sigma_chi, MF_cost, sigma_cost, f_spike, sigma_spike, cost_min, Bq, n=1): 
    
    """
    Helper function to draw chi-effective values corresponding to a component spin
    distribution with given parameters

    Parameters
    ----------
    mu_chi : float
        mean of spin magnitude beta distribution
    sigma_chi : float 
        std. dev. of spin magnitude beta distribution 
    MF_cost : float
        mixing fraction in aligned spin subpopulation for cos tilt angle distribution
    sigma_cost : float
        std. dev. of aligned spin subpopulation for cos tilt angle distribution
    frac_in_spike : float or None
        mixing fraction in half gaussian spike at spin mag. = 0
        (pass None if just looking at a distribution with a beta chi dist, not betaSpike)
    sigma_spike : float or None
        std. dev. of half gaussian spike at spin mag. = 0
        (pass None if just looking at a distribution with a beta chi dist, not betaSpike)
    cost_min : float
        lower truncation bound on the cosine tilt angle distribution
    Bq : float
        power law slope of the mass ratio distribution 
    n : int
        number of chi-eff samples to draw from the distribution

    Returns
    -------
    med : float
        Median of samples
    upperError : float
        Difference between 95th and 50th percentiles of data
    lowerError : float
        Difference between 50th and 5th percentiles of data
    """
    
    # transform from mu and sigma to a and b for beta distribution
    a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2)
    
    # draw uniform component spins + masses
    nRandomDraws = 10000
    samp_idxs = np.arange(nRandomDraws)
    chi1s = np.random.rand(nRandomDraws)
    chi2s = np.random.rand(nRandomDraws)
    cost1s = np.random.rand(nRandomDraws)*2 - 1
    cost2s = np.random.rand(nRandomDraws)*2 - 1
    mAs = np.random.rand(nRandomDraws)*100
    mBs = np.random.rand(nRandomDraws)*100
    m1s = np.maximum(mAs, mBs)
    m2s = np.minimum(mAs, mBs)
    
    # calculate p(spins,masses) for these uniform samples,
    # using functions from posterior_helper_functions.py
    if f_spike is None: 
        p_chi1 = betaDistribution(chi1s, a, b)
        p_chi2 = betaDistribution(chi2s, a, b)
    else: 
        p_chi1 = betaDistributionPlusSpike(chi1s, a, b, f_spike, sigma_spike)
        p_chi2 = betaDistributionPlusSpike(chi2s, a, b, f_spike, sigma_spike)
    p_cost1 = calculate_Gaussian_Mixture_1D(cost1s, 1, sigma_cost, MF_cost, cost_min, 1)
    p_cost2 = calculate_Gaussian_Mixture_1D(cost2s, 1, sigma_cost, MF_cost, cost_min, 1)
    p_masses = p_astro_masses(m1s, m2s, bq=Bq)
    
    weights = p_chi1*p_chi2*p_cost1*p_cost2*p_masses
    weights_normed = weights/np.sum(weights)
    weights_normed[np.where(weights_normed<0)] = 0 # get rid of tiny division errors
    
    # select a subset of the samples subject to the weights
    # calculated from p(spins,masses)
    idxs = np.random.choice(samp_idxs, p=weights_normed, size=n)  
    
    # calculate chi-eff for these samples
    q = m2s[idxs]/m1s[idxs]
    chi_eff = (chi1s[idxs]*cost1s[idxs] + q*chi2s[idxs]*cost2s[idxs])/(1+q)
    
    return chi_eff
