
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np


def score_func(reference, prediction, n_slices=1000):
    score = 0
    np.random.seed(0)
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


def compute_grad_norm(module, d=2):
    grads = [p.grad.norm(d).item() for p in filter(lambda p: p.grad is not None, module.parameters())]
    return np.mean(grads)


def plot_hist(x, y):
    fig = plt.figure(figsize=(20,10));
    ax = fig.subplots(1);
    _, bins, _ = ax.hist(x,bins=100, label='true', alpha=0.7, color='green');
    _ = ax.hist(y, bins=bins, label='fake', alpha=0.7, color='yellow');
    fig.legend();
    plt.close(fig)
    return fig


class TqdmMock:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


class NoGrad:
    def __init__(self, *modules):
        self.modules = modules

    def __enter__(self):
        for m in self.modules:
            for p in m.parameters():
                p.requires_grad = False

    def __exit__(self, type, value, traceback):
        for m in self.modules:
            for p in m.parameters():
                p.requires_grad = True