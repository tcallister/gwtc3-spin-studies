import numpy as np

def downsample(sampleDict_full, eventsMore, nBaseline=8000, nMore=50000): 
    
    """
    Function to draw the initial walkers for emcee from a uniform distribution
    
    Parameters
    ----------
    sampleDict_full : dict 
        Precomputed dictionary containing posterior samples for each event in our catalog
    eventsMore : list of strings
        Names of events for which we want to include nMore samples
    nBaseline : int
        Baseline number of samples to include for each event
    nMore : int 
        Number of samples to include for the events specifies in eventsMore
        
    Returns
    -------
    sampleDict : dict
        Dictionary with downsampled posterior samples for each event in sampleDict_full
    """
    
    
    sampleDict = {}
    
    # Cycle through events
    for event in sampleDict_full: 
        
        # Fetch full number of samples for this event
        nSamples_total = sampleDict_full[event]['a1'].size
        
        # Figure out what number to downsample to based off of whether the event is in 
        # eventsMore and whether the event has too few samples to downsample to the desired
        # number
        if event in eventsMore: 
            nDownsample = min(nSamples_total, nMore)
        else: 
            nDownsample = min(nSamples_total, nBaseline)
         
        # Choose random indices and downsample
        idxs = np.random.choice(nSamples_total, size=nDownsample, replace=False)
        sampleDict[event] = {var:sampleDict_full[event][var][idxs] for var in sampleDict_full[event]}
    
    return sampleDict 