# -*- coding: utf-8 -*-

import numpy as np


def get_abs_error(target, prediction):
    '''
    compute the  absolute_error

    Parameters:
    ----------------
    target:np.array
    prediction: np.array
    '''

    assert (target.shape == prediction.shape)

    return np.mean(np.abs(target - prediction))


def get_rsme_error(target, prediction):
    '''
    compute the  absolute_error

    Parameters:
    ----------------
    target: np.array
    prediction: np.array
    '''

    assert (target.shape == prediction.shape)

    return np.sqrt(np.mean(np.square(target - prediction)))



print('mae={}'.format(get_abs_error(original_reward.flatten(), adv_reward.flatten())))
print('RSME:{0}'.format(get_rsme_error(original_reward.flatten(), adv_reward.flatten())))
corr_matrix = np.corrcoef(original_reward.flatten()+1e-5, adv_reward.flatten())
print('corr of pred and gt:{}'.format(corr_matrix[0, 1]))