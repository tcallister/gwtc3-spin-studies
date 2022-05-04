import numpy as np
import glob
import emcee as mc
import json
import sys
from posterior_helper_functions import draw_initial_walkers_uniform
from posteriors import betaSpikePlusTruncatedMixture
from postprocessing import processEmceeChain 
from downsample_sampleDict import downsample

# set seed for reproducibility (number chosen arbitrarily)
np.random.seed(2647)

"""
Definitions and loading data
"""

# Model 
model = "component_spin_betaSpikePlusTruncatedMixture"

# Repository root 
froot = "/home/simona.miller/gwtc3-spin-studies/"

# Define emcee parameters
nWalkers = 20       # number of walkers 
dim = 8             # dimension of parameter space (number hyper params)
nSteps = 40000      # number of steps for chain

# Set prior bounds (where applicable, same as Table XII in https://arxiv.org/pdf/2111.03634.pdf)
priorDict = {
    'mu_chi':(0., 1.),
    'sigma_chi':(0.07, 0.5), # the sigma^2 bounds are [0.005, 0.25], so sigma goes from [sqrt(0.005), 4]  
    'MF_cost':(0., 1.),
    'sigma_cost':(0.1, 4.), 
    'frac_in_spike':(0, 1.), 
    'sigma_spike':(0.02, 0.1), 
    'cost_min':(-1,1)
}

# Load sampleDict
sampleDict_full = np.load(froot+"code/input/sampleDict_FAR_1_in_1_yr.pickle", allow_pickle=True)

# Get rid of non BBH events
non_BBHs = ['GW170817','S190425z','S190426c','S190814bv','S190917u','S200105ae','S200115j']
for event in non_BBHs:
    sampleDict_full.pop(event)
    
# Downsample events
events_more = [ # events that we want more samples for -- see downsample documentation
    'S191109d', 'S191103a', 'S190728q', 'GW170729', 'S190519bj', 'S190620e', 
    'S190805bq', 'S190517h', 'S190412m', 'GW151226', 'S191204r', 'S190719an', 
    'S190521g', 'S191127p', 'S200128d', 'S190706ai', 'S190720a', 'S190929d', 
    'S190527w', 'S200225q', 'S200129m', 'S191216ap', 'S190828j', 'S200216br', 
    'S190602aq', 'S200224ca', 'S190925ad', 'S200209ab'
]
sampleDict = downsample(sampleDict_full, events_more)

# Load injectionDict
injectionDict = np.load(froot+"code/input/injectionDict_FAR_1_in_1.pickle", allow_pickle=True)

# Will save emcee chains temporarily in the .tmp folder in this directory
output_folder_tmp = froot+"code/emceeCode/.tmp/"
output_tmp = output_folder_tmp+model


"""
Initializing emcee walkers or picking up where an old chain left off
"""

# Search for existing chains
old_chains = np.sort(glob.glob("{0}_r??.npy".format(output_tmp)))

# If no chain already exists, begin a new one
if len(old_chains)==0:
    
    print('\nNo old chains found, generating initial walkers ... ')

    run_version = 0

    # Initialize walkers
    initial_mu_chis = draw_initial_walkers_uniform(nWalkers, (0.2,0.4))
    initial_sigma_chis = draw_initial_walkers_uniform(nWalkers, (0.17,0.25))
    initial_MFs = draw_initial_walkers_uniform(nWalkers, (0.5,1.0))
    initial_sigma_costs = draw_initial_walkers_uniform(nWalkers, (0.1, 2.))
    initial_fracs = draw_initial_walkers_uniform(nWalkers, (0.4, 0.6))
    initial_sigma_ss = draw_initial_walkers_uniform(nWalkers, (0.05,0.07))
    initial_min_costs = draw_initial_walkers_uniform(nWalkers, (-0.7,-0.3))
    initial_Bqs = np.random.normal(loc=0, scale=3, size=nWalkers)
    
    # Put together all initial walkers into a single array
    initial_walkers = np.transpose(
        [initial_mu_chis, initial_sigma_chis, initial_MFs, initial_sigma_costs,
         initial_fracs, initial_sigma_ss,initial_min_costs,initial_Bqs]
    )
            
# Otherwise resume existing chain
else:
    
    print('\nOld chains found, loading and picking up where they left off ... ' )
    
    # Load existing file and iterate run version
    old_chain = np.concatenate([np.load(chain, allow_pickle=True) for chain in old_chains], axis=1)
    run_version = int(old_chains[-1][-6:-4])+1

    # Strip off any trailing zeros due to incomplete run
    goodInds = np.where(old_chain[0,:,0]!=0.0)[0]
    old_chain = old_chain[:,goodInds,:]

    # Initialize new walker locations to final locations from old chain
    initial_walkers = old_chain[:,-1,:]
    
    # Figure out how many more steps we need to take 
    nSteps = nSteps - old_chain.shape[1]
    
        
print('Initial walkers:')
print(initial_walkers)


"""
Launching emcee
"""

if nSteps>0: # if the run hasn't already finished

    assert dim==initial_walkers.shape[1], "'dim' = wrong number of dimensions for 'initial_walkers'"

    print(f'\nLaunching emcee with {dim} hyper-parameters, {nSteps} steps, and {nWalkers} walkers ...')

    sampler = mc.EnsembleSampler(
        nWalkers,
        dim,
        betaSpikePlusTruncatedMixture, # model in posteriors.py
        args=[sampleDict,injectionDict,priorDict], # arguments passed to betaPlusMixture
        threads=16
    )

    print('\nRunning emcee ... ')

    for i,result in enumerate(sampler.sample(initial_walkers,iterations=nSteps)):

        # Save every 10 iterations
        if i%10==0:
            np.save("{0}_r{1:02d}.npy".format(output_tmp,run_version),sampler.chain)

        # Print progress every 100 iterations
        if i%100==0:
            print(f'On step {i} of {nSteps}', end='\r')

    # Save raw output chains
    np.save("{0}_r{1:02d}.npy".format(output_tmp,run_version),sampler.chain)


"""
Running post processing and saving results
"""

print('\nDoing post processing ...')

if nSteps>0: 

    # If this is the only run, just process this one directly 
    if run_version==0:
        chainRaw = sampler.chain

    # otherwise, put chains from all previous runs together 
    else:
        previous_chains = [np.load(chain, allow_pickle=True) for chain in old_chains]
        previous_chains.append(sampler.chain)
        chainRaw = np.concatenate(previous_chains, axis=1)

else: 
    chainRaw = old_chain

# Run post-processing
chainDownsampled = processEmceeChain(chainRaw) 

# Format output into an easily readable format 
results = {
    'mu_chi':{'unprocessed':chainRaw[:,:,0].tolist(), 'processed':chainDownsampled[:,0].tolist()},
    'sigma_chi':{'unprocessed':chainRaw[:,:,1].tolist(), 'processed':chainDownsampled[:,1].tolist()},
    'MF_cost':{'unprocessed':chainRaw[:,:,2].tolist(), 'processed':chainDownsampled[:,2].tolist()},
    'sigma_cost':{'unprocessed':chainRaw[:,:,3].tolist(), 'processed':chainDownsampled[:,3].tolist()},
    'frac_in_spike':{'unprocessed':chainRaw[:,:,4].tolist(), 'processed':chainDownsampled[:,4].tolist()},
    'sigma_spike':{'unprocessed':chainRaw[:,:,5].tolist(), 'processed':chainDownsampled[:,5].tolist()},
    'cost_min':{'unprocessed':chainRaw[:,:,6].tolist(), 'processed':chainDownsampled[:,6].tolist()},
    'Bq':{'unprocessed':chainRaw[:,:,7].tolist(), 'processed':chainDownsampled[:,7].tolist()},
} 

# Save
savename = froot+"data/{0}.json".format(model)
with open(savename, "w") as outfile:
    json.dump(results, outfile)
print(f'Done! Run saved at {savename}')