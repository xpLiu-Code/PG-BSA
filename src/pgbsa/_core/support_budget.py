"""Deterministic plateau collapse and explicit initial-route budget."""
import numpy as np


def bounded_peak_indices(indices, logits, maximum):
    indices=np.sort(np.asarray(indices,dtype=np.int64))
    values=np.asarray(logits)
    if maximum < 1:
        raise ValueError('The route budget must be positive')
    if not len(indices):
        return indices
    joined=(np.diff(indices)==1)&(values[indices[1:]]==values[indices[:-1]])
    boundaries=np.r_[0,np.flatnonzero(~joined)+1,len(indices)]
    representatives=np.array([indices[(a+b-1)//2] for a,b in zip(boundaries[:-1],boundaries[1:])])
    order=np.argsort(-values[representatives],kind='stable')[:maximum]
    return np.sort(representatives[order])
