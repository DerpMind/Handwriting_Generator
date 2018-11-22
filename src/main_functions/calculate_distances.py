from scipy.spatial import distance_matrix
import numpy as np

def aggregate(encodings, letter_weights=None):
    '''Takes all letter encodings over all fonts and latent dimensions as a multi-dimensional array
    with dimensionality l * f * d with
    l: number of letters
    f: number of fonts
    d: number of latent dimensions ("encodings")
    
    letter weights carry information of how strongly each letter is to be valued for the aggregate distance measure
    
    Returns aggregate distance matrix between fonts (pairwise)
    with dimensionality f * f
    
    '''
    
    # initialize the distance matrix
    f = encodings.shape[1]
    l = encodings.shape[0]

    #TODO: Insert a weights vector
    if letter_weights==None:
        letter_weights = np.ones(l)    

    # calculate the individual distance matrices and aggregate them over all letters
    distances = np.zeros((f,f))
    for idx, weight in enumerate(letter_weights):
        distances = distances + \
                    weight * distance_matrix(encodings[idx,:,:],encodings[idx,:,:])
    return distances