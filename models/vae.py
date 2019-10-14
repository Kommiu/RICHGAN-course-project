import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange, tqdm
import torch.utils.data as data


from lib.utils import plot_hist, score_func
import lib.collections as lc


class Encoder(nn.Module):
    def __init__(
            self,
            latent_dim,
            condition_dim,
            hidden_dims,
            target_dim,
            activation=nn.LeakyReLU(0.2, inplace=True)
    ):
        super().__init__()
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(activation)
            return nn.Sequential(*layers)

        self.net = nn.Sequential()
        self.net.add_module(
            'block_0',
            block(target_dim + condition_dim, hidden_dims[0], False)
        )
        if len(hidden_dims) > 1:
            for i, (in_dim, out_dim) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:]), 1):
                self.net.add_module(
                    f'block_{i}',
                    block(in_dim, out_dim)
                )

        self.linear_means = nn.Linear(hidden_dims[-1], latent_dim)
        self.linear_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, X, C):
        inputs = torch.cat([X.squeeze(), C.squeeze()], dim=-1)
        features = self.net(inputs)
        means = self.linear_means(features)
        logvars = self.linear_logvar(features)
        return means, logvars


class Decoder(nn.Module):
    def __init__(
            self,
            target_dim,
            latent_dim,
            condition_dim,
            hidden_dims,
            activation=nn.LeakyReLU(0.2, inplace=True)
    ):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module(
            'fc_0',
            nn.Sequential(
                nn.Linear(latent_dim + condition_dim, hidden_dims[0]),
                activation
            )

        )
        if len(hidden_dims) > 1:
            for i, (in_dim, out_dim) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:]), 1):
                self.net.add_module(
                    f'fc_{i}',
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        activation,
                    )
                )
        self.net.add_module(
            'logits', nn.Linear(hidden_dims[-1], target_dim)
        )

    def forward(self, Z, C):
        inputs = torch.cat([Z.squeeze(), C.squeeze()], dim=-1)
        return self.net(inputs)


class CVAE:
    def __init__(self, encoder, decoder, optimizer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.device = next(self.encoder.parameters()).device

    @staticmethod
    def criterion(recon_x, x, means, logvars):
        recon = F.mse_loss(recon_x, x, reduction='sum')
        # recon = F.binary_cross_entropy_with_logits(recon_x, x)
        kld = -0.5 * torch.sum(1 + logvars - means.pow(2) - logvars.exp())
        return recon + kld

    @staticmethod
    def make_dataloader(targets, conditions, batch_size, num_workers):
        dataset = lc.Dataset(conditions, targets)
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    def train(
        self,
        dataloaders,
        start_epoch=0,
        num_epochs=100,
        writer=None,
        test_freq=1,
        plot_dists=False,
        **kwargs,
    ):
        epoch_tqdm = trange(start_epoch, start_epoch + num_epochs)
        batch_tqdm = {
            phase: tqdm(total=len(dataloaders['train']), desc=f'{phase}')
            for phase in dataloaders
        }
        for epoch in epoch_tqdm:
            loss_sum = 0
            loss_count = 0

            for i, (c, x) in enumerate(dataloaders['train']):

                x = x.float().to(self.device)
                c = c.float().to(self.device)
                b_size = len(x)
                means, logvars = self.encoder(x, c)
                std = torch.exp(0.5*logvars)

                eps = torch.randn(b_size, self.encoder.latent_dim, dtype=torch.float, device=self.device)
                z = eps * std + means
                recon_x = self.decoder(z, c)

                loss = self.criterion(recon_x, x, means, logvars)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()*b_size
                loss_count += b_size
                batch_tqdm['train'].update()

            score = None
            figs = None
            if 'test' in dataloaders and (epoch - start_epoch) % test_freq == 0:
                score, figs = self.test(dataloaders['test'], tqdm=batch_tqdm['test'], plot_dists=plot_dists)

            if writer is not None:
                writer.add_scalar('loss', loss_sum /loss_count, global_step=epoch)
                if score is not None:
                    writer.add_scalar('KS-score', score, global_step=epoch)
                if figs is not None:
                    for i, fig in enumerate(figs):
                        writer.add_figure(f'column-{i}', fig, global_step=epoch)
            for t in batch_tqdm.values():
                t.reset()

    def test(self, dataloader, tqdm=None, plot_dists=False, n_slises=300):
        reference = []
        generated = []
        with torch.no_grad():
            for c, x in dataloader:
                b_size = len(x)
                reference.append(torch.cat([x, c], dim=-1).squeeze())
                z = torch.randn(b_size, self.encoder.latent_dim, dtype=torch.float, device=self.device)
                x_prime = self.decoder(z, c.to(self.device)).cpu()
                generated.append(torch.cat([x_prime, c.squeeze()], dim=-1))
                if tqdm is not None:
                    tqdm.update()

        reference = torch.cat(reference, dim=0).numpy()
        generated = torch.cat(generated, dim=0).numpy()

        if plot_dists:
            figs = list()
            for i in range(x.size(1)):
                figs.append(plot_hist(reference[:, i], generated[:, i]))
        else:
            figs = None

        return score_func(reference, generated, n_slises), figs

    def generate(self, C):

        Z = torch.randn(len(C), self.encoder.latent_dim, device=self.device, dtype=torch.float)
        C = torch.from_numpy(C).float().to(self.device)
        with torch.no_grad():
            X = self.decoder(Z, C)
        return X.cpu().numpy()
