import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

class DiscreteGauss:

    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.binarizer = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense')

    def make_dataloader(self, X, C, *args, **kwargs):
        return X, C

    def train(self, dataloaders,  *args, **kwargs):
        x, c = dataloaders['train']
        n_features = c.shape[1]
        self.ohe_cols = [
            '{}_{}'.format(a, b + 1)
            for a, b
            in zip(np.repeat(range(n_features), self.n_bins), np.tile(range(self.n_bins), n_features))
            ]
        labels = self.binarizer.fit_transform(c)
        labels = pd.DataFrame(labels, columns=self.ohe_cols, dtype=np.int8)

        train = pd.concat([labels, pd.DataFrame(x)], axis=1)
        self.gmean = x.mean()
        self.gstd = x.std()

        self.means = train.groupby(self.ohe_cols, as_index=False).mean()
        self.stds = train.groupby(self.ohe_cols, as_index=False).std(ddof=0)

    def generate(self, c):
        labels = self.binarizer.transform(c)
        labels = pd.DataFrame(labels, columns=self.ohe_cols, dtype=np.int8)

        means = pd.merge(labels, self.means, how='left').drop(self.ohe_cols, axis=1).fillna(self.gmean)
        stds = pd.merge(labels, self.stds, how='left').drop(self.ohe_cols, axis=1).fillna(self.gstd)

        pred = np.random.normal(loc=means, scale=stds)

        return pred
