import torch.nn as nn
import pandas as pd
import numpy as np


from submissions.model_1 import model, condition_transformer, target_transformer, config


class Model():
    def __init__(self, simulate_error_codes=True):

        self.__dict__ = locals()
        self.model = model
        self.condition_transformer = condition_transformer
        self.target_transformer = target_transformer
        self.simulate_error_codes = simulate_error_codes
        self.condition_columns = ['TrackP', 'TrackEta', 'NumLongTracks']

    def fit(self, conditions, targets):
        self.target_columns = targets.columns
        conditions = conditions[self.condition_columns]
        if self.simulate_error_codes:
            mask1 = (targets == -999).values.all(axis=1)
            mask2 = (targets == 0).values.all(axis=1)
            mask = (mask1 | mask2)

            targets = targets[~mask]
            conditions = conditions[~mask]

            self.probs1 = mask.mean()
            self.probs2 = mask1.mean() / mask.mean()

        X = self.target_transformer.fit_transform(targets.values)
        C = self.condition_transformer.fit_transform(conditions.values)

        dataloaders = {'train': self.model.make_dataloader(X, C, **config['dataloader'])}
        self.model.train(dataloaders, **config['train'])

    def predict(self, conditions):
        C = self.condition_transformer.transform(conditions.values)
        X = self.model.generate(C).cpu().numpy()
        X = self.target_transformer.inverse_transform(X)
        if self.simulate_error_codes:
            step1 = np.random.binomial(1, self.probs1, (len(X), 1))
            step2 = np.random.binomial(1, self.probs2, (len(X), 1))
            X = -999 * step1 * step2 + (1 - step1) * X
        return pd.DataFrame(X, columns=self.target_columns, index=conditions.index)



