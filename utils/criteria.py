import numpy as np

# Random sampling
def random_sampling(y_pred_prob, n_samples):
    return np.random.choice(range(len(y_pred_prob)), n_samples)

# Rank all the unlabeled samples in an ascending order according to the least confidence
def least_confidence(y_pred_prob, n_samples):
    """
    y_pred_prob: (N, 5, 62)
    """
    origin_index = np.arange(0, len(y_pred_prob)) # (N)
    # max_prob = np.max(y_pred_prob, axis=1) # (N, 62)
    max_prob = np.max(y_pred_prob, axis=2) # (N, 5)
    mean_prob = np.mean(max_prob, axis=1) # (N,)
    # pred_label = np.argmax(y_pred_prob, axis=1) # (N, 62)
    # pred_label = np.argmax(y_pred_prob, axis=2) # (N, 5)

    # lci = np.column_stack((origin_index,
    #                        max_prob,
    #                        pred_label))
    lci = np.column_stack((origin_index,
                       mean_prob))
    lci = lci[lci[:, 1].argsort()]
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]

# Rank all the unlabeled samples in an ascending order according to the margin sampling
def margin_sampling(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    # margim_sampling = np.diff(-np.sort(y_pred_prob)[:, ::-1][:, :2]) # (N, 1)
    margim_sampling = np.diff(-np.sort(y_pred_prob)[:, :, ::-1][:, :, :2]) # (N, 5, 1)
    margim_sampling_mean = np.mean(margim_sampling, axis=1) # (N, 1)
    # pred_label = np.argmax(y_pred_prob, axis=1)
    # msi = np.column_stack((origin_index,
    #                        margim_sampling,
    #                        pred_label))
    msi = np.column_stack((origin_index,
                           margim_sampling_mean))
    msi = msi[msi[:, 1].argsort()]
    return msi[:n_samples], msi[:, 0].astype(int)[:n_samples]

# Rank all the unlabeled samples in an descending order according to their entropy
def entropy(y_pred_prob, n_samples):
    # entropy = stats.entropy(y_pred_prob.T)
    # entropy = np.nan_to_num(entropy)
    origin_index = np.arange(0, len(y_pred_prob))
    # entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1) # (4,)
    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=2) # (4, 5)
    mean_entropy = np.mean(entropy, axis=1) # (4,)
    
    # pred_label = np.argmax(y_pred_prob, axis=1)
    # eni = np.column_stack((origin_index,
    #                        entropy,
    #                        pred_label))
    eni = np.column_stack((origin_index,
                           mean_entropy))

    eni = eni[(-eni[:, 1]).argsort()]
    return eni[:n_samples], eni[:, 0].astype(int)[:n_samples]

def bvsb(y_pred_prob, n_samples):
    """
    y_pred_prob: (N, 5, 62)
    """
    origin_index = np.arange(0, len(y_pred_prob)) # (N)
    # max_prob = np.max(y_pred_prob, axis=1) # (N, 62)
    max1_prob = np.max(y_pred_prob, axis=2) # (N, 5)
    y_pred_prob[y_pred_prob==np.max(y_pred_prob,axis=2)[:, :, None]] = 0
    max2_prob = np.max(y_pred_prob, axis=2)
    max_prob = max2_prob/(max1_prob+1e-6)
    mean_prob = np.mean(max_prob, axis=1) # (N,)
    # pred_label = np.argmax(y_pred_prob, axis=1) # (N, 62)
    # pred_label = np.argmax(y_pred_prob, axis=2) # (N, 5)

    # lci = np.column_stack((origin_index,
    #                        max_prob,
    #                        pred_label))
    lci = np.column_stack((origin_index,
                       mean_prob))
    lci = lci[lci[:, 1].argsort()]
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]

def pmes(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob)) # (N)
    # max_prob = np.max(y_pred_prob, axis=1) # (N, 62)
    max1_prob = np.max(y_pred_prob, axis=2) # (N, 5)
    min_prob = np.min(y_pred_prob, axis=2) # (N, 5)
    y_pred_prob[y_pred_prob==np.max(y_pred_prob,axis=2)[:, :, None]] = 0
    max2_prob = np.max(y_pred_prob, axis=2)
    
    max_prob = 2*max1_prob-max2_prob-min_prob
    mean_prob = np.mean(max_prob, axis=1) # (N,)
    # pred_label = np.argmax(y_pred_prob, axis=1) # (N, 62)
    # pred_label = np.argmax(y_pred_prob, axis=2) # (N, 5)

    # lci = np.column_stack((origin_index,
    #                        max_prob,
    #                        pred_label))
    lci = np.column_stack((origin_index,
                       mean_prob))
    lci = lci[lci[:, 1].argsort()]
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]
