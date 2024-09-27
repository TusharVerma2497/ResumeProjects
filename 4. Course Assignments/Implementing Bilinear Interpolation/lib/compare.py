import numpy as np

def rms_distance(array1, array2):
    squared_diff = (array1 - array2) ** 2
    mean_squared_diff = squared_diff.mean()
    rms_dist = np.sqrt(mean_squared_diff)
    return rms_dist


def L1_distance(array1, array2):
    diff = np.abs(array1 - array2)
    mean_diff = diff.mean()
    return mean_diff


def getChangeMask(array1, array2, threshold=None, metric='L1'):
    if metric=='RMS':
        diff = (array1 - array2) ** 2
        diff = np.sqrt(diff)
    else:
        diff = np.abs(array1 - array2)

    channel_diff = np.mean(diff, axis=-1)
    if threshold==None:
        return 255-channel_diff
    
    # Filter the indices where the difference is greater than the threshold
    indices = np.where(channel_diff > threshold)
    mask=np.ones(shape=(array1.shape[0],array1.shape[1]))
    mask[indices]=channel_diff[indices]
    return 255-mask