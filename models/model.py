import pandas as pd
import numpy as np


class Model:
    def __init__(
            self,
            model,
            condition_transformer,
            target_transformer,
            simulate_error_codes=True,
    ):

        self.__dict__ = locals()
        self.model = model
        self.condition_transformer = condition_transformer
        self.target_transformer = target_transformer
        self.simulate_error_codes = simulate_error_codes
        self.condition_columns = ['TrackP', 'TrackEta', 'NumLongTracks']

    def fit(
            self,
            conditions,
            targets,
            start_epoch=0,
            num_epochs=40,
            n_critic=1,
            batch_size=512,
            writer=None,
            num_workers=4,

    ):

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

        dataloaders = {
            'train': self.model.make_dataloader(X, C, batch_size=batch_size, num_workers=num_workers)
        }
        self.model.train(
            dataloaders,
            start_epoch=start_epoch,
            num_epochs=num_epochs,
            n_critic=n_critic,
            writer=writer,
            log_grad_norms=True,
            plot_dists=True,

        )

    def predict(self, conditions):
        conditions = conditions[self.condition_columns]
        C = self.condition_transformer.transform(conditions.values)
        X = self.model.generate(C)
        X = self.target_transformer.inverse_transform(X)
        if self.simulate_error_codes:
            step1 = np.random.binomial(1, self.probs1, (len(X), 1))
            step2 = np.random.binomial(1, self.probs2, (len(X), 1))
            X = -999 * step1 * step2 + (1 - step1) * X

        result = pd.DataFrame(X, columns=self.target_columns, index=conditions.index)
        return result


