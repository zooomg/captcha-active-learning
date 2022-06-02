import numpy as np

# Random sampling
def random_sampling(y_pred_prob, n_samples):
    return np.random.choice(range(len(y_pred_prob)), n_samples)

# Rank all the unlabeled samples in an ascending order according to the least confidence
def least_confidence(y_pred_prob, n_samples, option):
    """
    y_pred_prob: (N, 5, 62)
    """
    origin_index = np.arange(0, len(y_pred_prob)) # (N)

    max_prob = np.max(y_pred_prob, axis=2) # (N, 5)
    if option=='mean':
        prob = np.mean(max_prob, axis=1) # (N,)
    elif option=='median':
        prob = np.median(max_prob, axis=1)
    else:
        raise Exception("Not proper option error, (mean, median)")
    lci = np.column_stack((origin_index, prob))
    lci = lci[lci[:, 1].argsort()]
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]

# Rank all the unlabeled samples in an ascending order according to the margin sampling
def margin_sampling(y_pred_prob, n_samples, option):
    origin_index = np.arange(0, len(y_pred_prob))

    margim_sampling = np.diff(-np.sort(y_pred_prob)[:, :, ::-1][:, :, :2]) # (N, 5, 1)
    
    if option=='mean':
        margins = np.mean(margim_sampling, axis=1) # (N, 1)
    elif option=='median':
        margins = np.median(margim_sampling, axis=1)

    msi = np.column_stack((origin_index, margins))
    msi = msi[msi[:, 1].argsort()]
    return msi[:n_samples], msi[:, 0].astype(int)[:n_samples]

# Rank all the unlabeled samples in an descending order according to their entropy
def entropy(y_pred_prob, n_samples, option):
    origin_index = np.arange(0, len(y_pred_prob))

    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=2) # (4, 5)
    
    if option=='mean':
        entropys = np.mean(entropy, axis=1) # (4,)
    elif option=='median':
        entropys = np.median(entropy, axis=1) # (4,)
        
    eni = np.column_stack((origin_index, entropys))
    eni = eni[(-eni[:, 1]).argsort()]
    return eni[:n_samples], eni[:, 0].astype(int)[:n_samples]

def bvsb(y_pred_prob, n_samples, option):
    """
    y_pred_prob: (N, 5, 62)
    """
    origin_index = np.arange(0, len(y_pred_prob)) # (N)

    max1_prob = np.max(y_pred_prob, axis=2) # (N, 5)
    y_pred_prob[y_pred_prob==np.max(y_pred_prob,axis=2)[:, :, None]] = 0
    max2_prob = np.max(y_pred_prob, axis=2)
    bvsb_prob = max2_prob/(max1_prob+1e-6)
    
    if option=='mean':
        probs = np.mean(bvsb_prob, axis=1) # (N,)
    elif option=='median':
        probs = np.median(bvsb_prob, axis=1)

    lci = np.column_stack((origin_index, probs))
    lci = lci[lci[:, 1].argsort()]
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]

def pmes(y_pred_prob, n_samples, option):
    origin_index = np.arange(0, len(y_pred_prob)) # (N)
    
    max1_prob = np.max(y_pred_prob, axis=2) # (N, 5)
    min_prob = np.min(y_pred_prob, axis=2) # (N, 5)
    y_pred_prob[y_pred_prob==np.max(y_pred_prob,axis=2)[:, :, None]] = 0
    max2_prob = np.max(y_pred_prob, axis=2)
    
    pmes_prob = 2*max1_prob-max2_prob-min_prob
    
    if option=='mean':
        probs = np.mean(pmes_prob, axis=1) # (N,)
    elif option=='median':
        probs = np.median(pmes_prob, axis=1) # (N,)

    lci = np.column_stack((origin_index, probs))
    lci = lci[lci[:, 1].argsort()]
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]
