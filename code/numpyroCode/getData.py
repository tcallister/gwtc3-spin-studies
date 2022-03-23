import numpy as np

def getInjections(mMin=5.0,
    mMax=88.2,
    delta_m=4.9,
    kappa=2.7,
    alpha=-3.5,
    mu_m=33.6,
    sig_m=4.7,
    f_peak = 0.03):

    injectionFile = "/Users/tcallister/Documents/Repositories/o3b-pop-studies/code/injectionDict_10-20_directMixture_FAR_1_in_1.pickle"
    injectionDict = np.load(injectionFile,allow_pickle=True)

    m1_det = np.array(injectionDict['m1'])
    m2_det = np.array(injectionDict['m2'])
    z_det = np.array(injectionDict['z'])
    Xeff_det = np.array(injectionDict['Xeff'])
    Xp_det = np.array(injectionDict['Xp'])
    dVdz_det = np.array(injectionDict['dVdz'])
    inj_weights = np.array(injectionDict['weights_XeffOnly'])

    p_m1_det_pl = (1.+alpha)*m1_det**alpha/(mMax**(1.+alpha) - mMin**(1.+alpha))
    p_m1_det_peak = np.exp(-(m1_det-mu_m)**2./(2.*np.pi*sig_m**2))/np.sqrt(2.*np.pi*sig_m**2.)
    p_m1_det = f_peak*p_m1_det_peak + (1.-f_peak)*p_m1_det_pl
    p_m1_det[m1_det>mMax] = 0.
    p_m1_det[m1_det<mMin] = 0.

    smoothing = np.ones_like(m1_det)
    to_smooth = (m1_det>mMin)*(m1_det<delta_m)
    smoothing[to_smooth] = (np.exp(delta_m/m1_det[to_smooth] + delta_m/(m1_det[to_smooth]-delta_m)) + 1)**(-1.)
    p_m1_det *= smoothing

    p_z_det = dVdz_det*(1.+z_det)**(kappa-1)
    pop_reweight = p_m1_det*p_z_det*inj_weights
    injectionDict['pop_reweight'] = pop_reweight

    return injectionDict

def getSamples(mMin=5.,
    mMax=88.2,
    delta_m=4.9,
    kappa=2.7,
    alpha=-3.5,
    mu_m=33.6,
    sig_m=4.7,
    f_peak = 0.03):

    # Dicts with samples:
    sampleDict = np.load("/Users/tcallister/Documents/LIGO-Data/sampleDict_FAR_1_in_1_yr.pickle",allow_pickle=True)
    sampleDict.pop('S190814bv')

    for event in sampleDict:
        print(event,len(sampleDict[event]['m1']))

    nEvents = len(sampleDict)
    for event in sampleDict:

        m1 = np.array(sampleDict[event]['m1'])
        m2 = np.array(sampleDict[event]['m2'])
        Xeff = np.array(sampleDict[event]['Xeff'])
        
        p_m1_pl = (1.+alpha)*m1**alpha/(mMax**(1.+alpha) - mMin**(1.+alpha))
        p_m1_peak = np.exp(-(m1-mu_m)**2./(2.*np.pi*sig_m**2))/np.sqrt(2.*np.pi*sig_m**2.)
        p_m1 = f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl
        p_m1[m1>mMax] = 0.
        p_m1[m1<mMin] = 0.

        smoothing = np.ones_like(m1)
        to_smooth = (m1>mMin)*(m1<delta_m)
        smoothing[to_smooth] = (np.exp(delta_m/m1[to_smooth] + delta_m/(m1[to_smooth]-delta_m)) + 1)**(-1.)
        p_m1 *= smoothing
        
        sampleDict[event]['weights_over_priors'] = sampleDict[event]['weights']*p_m1/sampleDict[event]['Xeff_priors']
        inds_to_keep = np.random.choice(np.arange(m1.size),size=4000,replace=True)
        for key in sampleDict[event].keys():
            sampleDict[event][key] = sampleDict[event][key][inds_to_keep]

    print(len(sampleDict))
    print(sampleDict.keys())
    return sampleDict

if __name__=="__main__":
    getSamples()
