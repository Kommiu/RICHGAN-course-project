import numpy as np


def compute_grad_norm(module, d=2):
    grads = [p.grad.norm(d).item() for p in filter(lambda p: p.grad is not None, module.parameters())]
    return np.mean(grads)