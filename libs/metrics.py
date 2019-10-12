from scipy.stats import ks_2samp
import numpy as np


def score_func(reference, prediction, n_slices=100):
    score = 0
    w_normal = np.random.normal(size=(n_slices, len(reference.T)))

    # prediction = sample2.values
    for k in range(n_slices):
        score = max(score,
                    ks_2samp(
                        np.sum(w_normal[k] * reference, axis=1),
                        np.sum(w_normal[k] * prediction, axis=1)
                    )[0]
                    )
    return score
